import os
import sys
__base_dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__base_dir__)))
sys.path.insert(0, __base_dir__)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import default_collate
from torch.distributed.fsdp import CPUOffload
from transformers import AutoModelForCausalLM, HfArgumentParser
from accelerate import Accelerator
from accelerate import FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.utils import DynamoBackend

from megatron.training import get_tokenizer
from dataset_loader import load_dataset


debug = False
logger = get_logger(__name__, "DEBUG" if debug else "INFO")
# logger.remove()
# logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class Distributed:

  def __init__(self, backend="nccl"):
    self.backend = backend

  def __enter__(self):
    dist.init_process_group(self.backend)
    return dist

  def __exit__(self, exc_type, exc_value, traceback):
    dist.destroy_process_group()


def is_first_rank():
  if not dist.is_initialized():
    return True
  rank = dist.get_rank()
  return rank == 0


def not_first_rank():
  if not dist.is_initialized():
    return False
  rank = dist.get_rank()
  return rank != 0


def report_args(args):
  logger.info("-" * 25 + "Arguments" + "-" * 25)
  for key in vars(args):
    logger.info(f"{key:.<48}{getattr(args, key)}")
  logger.info("-" * 25 + "Arguments" + "-" * 25)


@torch.no_grad()
def compute_loss(model, data):
  """ Compute cross entropy loss of the sequence.
  """
  data = default_collate([data])
  model.eval()
  logits = model.forward(data["tokens"].cuda(),
    labels=data["labels"].cuda(),
    position_ids=data["position_ids"].cuda(),
    )[1].detach().cpu()
  return F.cross_entropy(logits.transpose(1, 2), data["labels"])



@torch.no_grad()
def compute_loss_batch(model, batch):
  """ Compute cross entropy loss of a batch of sequence.
  """
  data = default_collate(batch)
  model.eval()
  logits = model.forward(data["tokens"].cuda(),
    labels=data["labels"].cuda(),
    position_ids=data["position_ids"].cuda(),
    )[1].detach().cpu()
  return torch.mean(F.cross_entropy(logits.transpose(1, 2), data["labels"], reduction='none'), dim=1)


@torch.no_grad()
def compute_loss_batch_dist(model, batch):
  """ Compute cross entropy loss of a batch of sequence.
  """
  rank = dist.get_rank()
  world = dist.get_world_size()
  model.eval()
  ce_loss = torch.zeros(len(batch), dtype=torch.float32, device="cuda")
  # # Column wise slice
  micro_batch = batch[rank::world]
  # Row wise slice
  # micro_batch = batch[rank * world:(rank + 1) * world]
  # logger.info(f"{rank=}, {len(micro_batch)=}", main_process_only=False)
  if micro_batch:
    data = default_collate(micro_batch)
    logits = model.forward(data["tokens"].cuda(),
      labels=data["labels"].cuda(),
      position_ids=data["position_ids"].cuda(),
    )[1].detach()
    ce_loss_part = torch.mean(F.cross_entropy(logits.transpose(1, 2), data["labels"].cuda(), reduction='none'), dim=1)
    ce_loss[rank::world] = ce_loss_part
  dist.all_reduce(ce_loss, op=dist.ReduceOp.SUM)
  return ce_loss.to("cpu")


def batched(original, batch_size=8):
  batch = []
  for sample in tqdm(original,
              desc=f"Batched model inference across multiple GPUs",
              unit="batch",
              disable=not_first_rank(),
              ):
    batch.append(sample)
    if len(batch) == batch_size:
      yield batch
      batch = []
  if batch:
    yield batch


def save_batch(batch):
  tok = get_tokenizer()
  txt = [tok.detokenize(np.array(b["tokens"])) for b in tqdm(batch)]
  dataset_paths = [train.datasets[b["dataset_id"]].dataset_path for b in batch]
  df = pd.DataFrame({"text": txt, "dataset_path": dataset_paths})
  df.to_csv("exmaple.csv", encoding="utf-8")


train, val, test, batch = None, None, None, None


def run_load_dataset(megatron_ckpt_path):
  global train, val, test
  train, val, test = load_dataset(megatron_ckpt_path)


def get_batch(begin=32307360, end=32308800):
  return [train[i] for i in tqdm(range(begin, end), desc="Loading batch", unit="sample", disable=not_first_rank())]


def prepare_datasets(begin=32307360, end=32308800):
  global batch
  run_load_dataset()
  batch = get_batch(begin=begin, end=end)


def extract_batch_info(train, batch, samples, offset, losses_samples):
  tok = get_tokenizer()
  s_ds_index = [batch[i]["dataset_id"] for i in samples]
  s_dataset_sample_index = [train.dataset_sample_index[train.dataset_shuffle_index[i]] for i in samples + offset]
  s_dataset_pointer = [train.datasets[s_ds_index[i]] for i in range(len(samples))]
  s_gptds_sample_index = [s_dataset_pointer[i].sample_index[s_dataset_pointer[i].shuffle_index[s_dataset_sample_index[i]]][0] \
                         for i in range(len(samples))]
  s_document_index = [train.datasets[s_ds_index[i]].document_index[s_gptds_sample_index[i]] for i in range(len(samples))]
  s_document_index_internal = [train.datasets[s_ds_index[i]].dataset.index.document_indices[s_document_index[i]] \
                              for i in range(len(samples))]
  dataset_paths = [train.datasets[batch[i]["dataset_id"]].dataset_path for i in samples]
  txt = [tok.detokenize(np.array(batch[i]["tokens"])) for i in tqdm(samples, desc="Decode tokens", unit="sample")]
  df = pd.DataFrame({
      "dataset_id": s_ds_index,
      "dataset_sample_index": s_dataset_sample_index,
      "document_index": s_document_index,
      "internal_document_index": s_document_index_internal,
      "text": txt,
      "dataset_path": dataset_paths,
      "ce-loss": losses_samples,
      })
  return df


def get_losses_batch(model, batch, infer_batch=4, distributed=False):
  if distributed:
    world = dist.get_world_size()
    return torch.concat([compute_loss_batch_dist(model, b) \
          for b in batched(batch, batch_size=infer_batch * world)])
  elif infer_batch == 1:
    return np.array([compute_loss(model, b) for b in tqdm(batch, desc="Model inference", unit="sample")])
  else:
    return torch.concat([compute_loss_batch(model, b) \
        for b in batched(batch, batch_size=infer_batch)])


def run_worker(model, sample_indices, batch_size=1440, infer_batch=4, distributed=False):
  rank = dist.get_rank()
  if train is None:
    raise RuntimeError("Run run_load_dataset() first.")
  df_total = None
  for sample_end in tqdm(sample_indices, desc="Run through batches", unit="batch", disable=not_first_rank()):
    sample_begin = sample_end - batch_size
    logger.info(f"Getting batch of {batch_size} from {sample_begin} to {sample_end}.")
    batch = get_batch(sample_begin, sample_end)
    logger.info(f"Compute loss for each sample.")
    dist.barrier()
    losses_batch = get_losses_batch(model, batch, infer_batch=infer_batch, distributed=distributed)
    if is_first_rank():
      top5_samples = np.argsort(losses_batch)[-5:]
      losses_samples = losses_batch[top5_samples]
      logger.info(f"The 5 samples with highest losses: {top5_samples!s}.")
      logger.info(f"Extract metadata of these samples from the batch.")
      df = extract_batch_info(train, batch, top5_samples, sample_begin, losses_samples)
      df_total = pd.concat([df_total, df])
      dist.barrier()
    else:
      dist.barrier()
  return df_total


def save_cache_dist(rank, data, prefix):
  filename = f"{prefix}_{rank}.pt"
  dirname = os.path.dirname(filename)
  if is_first_rank():
    os.makedirs(dirname, exist_ok=True)
  torch.save(data, filename)
  

def load_cache_dist(rank, prefix):
  filename = f"{prefix}_{rank}.pt"
  if not os.path.exists(filename):
    return None
  return torch.load(filename)


@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Path to model in HF format."})
    megatron_ckpt_path: str = field(metadata={"help": "Megatron Model checkpoint."})
    input_file: str = field(metadata={"aliases": ["-i"], "help": "Input files containning consumed samples counts of the batches to inspect."})
    batch_size: int = field(metadata={"help": "Global batch size."})
    infer_batch: Optional[int] = field(default=4, metadata={"help": "Inference batch size"})
    distributed: Optional[bool] = field(default=False, metadata={"help": "Use distributed inference"})
    output_file: Optional[str] = field(default="spike-samples.xlsx", metadata={"help": "Output XLSX file."})
    # cache_train_dataset: Optional[str] = field(default=None, metadata={"help": "Cache train dataset to disk."})

def main():
  parser = HfArgumentParser((ScriptArguments))
  (script_args, ) = parser.parse_args_into_dataclasses()

  dist.init_process_group("nccl")
  rank = dist.get_rank()
  world = dist.get_world_size()
  acc = Accelerator(
          device_placement=True,
          mixed_precision='bf16',
          fsdp_plugin=FullyShardedDataParallelPlugin(
                  use_orig_params=True,
                  activation_checkpointing=True,
                  cpu_offload=CPUOffload(offload_params=True),
                  ),
          dynamo_backend=DynamoBackend.INDUCTOR,
  )
  logger.debug(f"rank = {dist.get_rank()}", main_process_only=False)
  report_args(script_args)
  #  base = "/mnt/cpfs/training/pretrain/output/lfu-14b-pretrain-v8-1/"
  logger.info(f"Load model")
  model = AutoModelForCausalLM.from_pretrained(
      script_args.model_path,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
      # device_map="cuda",
      # device_map="auto",
      )
  model = acc.prepare_model(model)
  logger.info(f"model: {model.device}", main_process_only=False)
  logger.info(f"Load dataset")
  run_load_dataset(script_args.megatron_ckpt_path)
  logger.info(f"Done loading dataset")

  # Samples at the end of each iteration where there is a spike
  # batches_of_interest = [ 25422720,  28628160,  32280000,  32307360,  32308800,  62164320,
  #                         62165760,  64825440,  71907360,  71911680,  71913120,  71918880,
  #                         74137920,  96344160, 110506560, 143119680, 143132640]
  batches_of_interest = np.loadtxt(script_args.input_file, dtype=int)

  df = run_worker(model, batches_of_interest, batch_size=script_args.batch_size, infer_batch=script_args.infer_batch, distributed=script_args.distributed)
  if is_first_rank():
    df.reset_index(inplace=True)
    torch.save(df, "%s.pt" % os.path.splitext(script_args.output_file)[0])
    if script_args.output_file.endswith("xlsx"):
      df.to_excel(script_args.output_file, engine="xlsxwriter")
    elif script_args.output_file.endswith("csv")
      df.to_csv(script_args.output_file, encoding="utf-8")
    else:
      logger.warn(f"Unsupported file format, will skip: {script_args.output_file}")

  dist.destroy_process_group()


if __name__ == "__main__":
  with logging_redirect_tqdm():
    main()
