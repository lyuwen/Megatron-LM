import os
import sys
__base_dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__base_dir__)))
sys.path.insert(0, __base_dir__)
import torch
import argparse
try:
  from loguru import logger
except ImportError:
  import logging
  logger = logging.getLogger(__name__)
  logging.basicConfig(encoding='utf-8', level=logging.INFO)

from megatron.training import get_tokenizer
from megatron.training.global_vars import set_global_variables
from megatron.training.arguments import _print_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder


def _check_megatron_path(path):
    return os.path.exists(f"{path}/megatron") and os.path.exists(f"{path}/pretrain_gpt.py")


def find_megatron_path(megatron_path):
    if megatron_path is not None:
        if _check_megatron_path(megatron_path):
            return megatron_path
        else:
            raise ValueError(f"Megatron not found in {megatron_path=}.")
    #1 Megatron lib is in current working directory
    megatron_path = os.getcwd()
    if _check_megatron_path(megatron_path):
        return megatron_path
    #2 Megatron lib is in the same directory as the script
    megatron_path = os.path.dirname(os.path.abspath(__file__))
    if _check_megatron_path(megatron_path):
        return megatron_path
    #3 Megatron lib is in the second outter directory from the script
    megatron_path = os.path.dirname(os.path.dirnamepath(megatron_path))
    if _check_megatron_path(megatron_path):
        return megatron_path
    raise ValueError(f"Megatron not found.")


def get_train_valid_test_num_samples(args):
    """Train/valid/test num samples."""

    # Number of train/valid/test samples.
    if args.reset_dataloader and args.force_train_samples:
        train_samples = (args.train_iters - args.iteration) * args.global_batch_size
        assert args.force_train_samples >= train_samples, \
            f"Input force-train-samples smaller than train iterations to run."
        train_samples = args.force_train_samples
    elif args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    if args.reset_dataloader:
        eval_iters = ((args.train_iters - args.iteration) // args.eval_interval + 1) * \
                     args.eval_iters
    else:
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
    test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )


def is_dataset_built_on_rank():
    return True


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        renormalize_blend_weights=args.renormalize_blend_weights,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path = args.s3_cache_path,
        use_distributed_builder = False,
        use_fast_blend_indices = True,
    )



def train_valid_test_datasets_provider(args, train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    config = core_gpt_dataset_config_from_args(args)

    dataset_type = GPTDataset

    print("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True, help="Checkpoint load directory.")
    parser.add_argument('--megatron-path', help='Megatron code path.')

    args = parser.parse_args(args)

    megatron_path = find_megatron_path(args.megatron_path)
    args.megatron_path = megatron_path
    logger.info(f"Megatron-Core library found in {megatron_path}, updating environ variable.")
    sys.path.insert(0, megatron_path)
    return args


def load_dataset(ckpt_dir):
    with open(get_checkpoint_tracker_filename(ckpt_dir), 'rt') as f:
        iteration = int(f.read())
    ckpt_base = get_checkpoint_name(ckpt_dir, iteration, False, 1, 0, 0, 1, 0, return_base_dir=True)
    data = torch.load(f"{ckpt_base}/common.pt")
    ckpt_args = data['args']
    _print_args("Checkpoint Arguments", ckpt_args)

    set_global_variables(ckpt_args)
    train_val_test_num_samples = get_train_valid_test_num_samples(ckpt_args)
    train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(ckpt_args, train_val_test_num_samples)
    return train_ds, valid_ds, test_ds


def main(args=None):
    args = parse_args(args)
    ckpt_dir = args.load
    return load_dataset(ckpt_dir)


if __name__ == "__main__":
    retval = main()
