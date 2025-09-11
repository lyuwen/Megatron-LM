# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
import numpy as np
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.gpt_dataset_mm import GPTDatasetMM
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    get_batch_on_this_tp_rank_sft,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
# LFu
from megatron.core.transformer.moe.utils import (
    get_moe_model_size,
    get_moe_activated_size,
    get_embedding_size,
    get_moe_FLOPs,
)
from megatron.core.datasets.json_sft import JSONSFTDataset, SFTPreTokenizedDataset

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 5680))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, getattr(args, "moe_use_legacy_grouped_gemm", False))
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, getattr(args, "moe_use_legacy_grouped_gemm", False))

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )
        if args.num_experts is not None:
            print_rank_0("-" * 18 + " Model  Summary " + "-" * 18)
            print_rank_0(f"Number of trainable parameters in the model (exclude embedding): {get_moe_model_size(args):,d}")
            print_rank_0(f"Number of activated parameters in the model (exclude embedding): {get_moe_activated_size(args):,d}")
            print_rank_0(f"Number of parameters in the embedding layer: {get_embedding_size(args):,d}")
            print_rank_0(f"Model Structure:\n{model!s}")
            print_rank_0("-" * 52)

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()

    #  if args.train_mode == "pretrain":
        #  raise ValueError('The JSON-SFT dataset should only be used for finetuning!')
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank_sft(data_iterator , per_seq_average=False)
    # slice batch along sequence dimension for context parallelism
    num_seqs = batch.pop('num_seqs')
    batch = get_batch_on_this_cp_rank(batch)

    return (
        batch['tokens'],
        batch['labels'],
        batch['loss_mask'],
        batch['attention_mask'],
        batch['position_ids'],
        num_seqs,
        None
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor,  num_seqs: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    if num_seqs is None:
        return (
            loss[0] * args.context_parallel_size,
            #  local_num_tokens,
            {'lm loss': (reporting_loss[0], reporting_loss[1])},
        )
    return (
        loss[0] * args.context_parallel_size,
        #  local_num_tokens,
        num_seqs.sum(),
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = \
            get_batch(data_iterator)
    timers("batch-generator").stop()
    
    # if torch.distributed.get_rank() == 0:   
    #     print("labels: ", labels.detach().cpu().tolist()[0][:1000])
    #     print("tokens: ", tokens.detach().cpu().tolist()[0][:1000])
    #     print("loss_mask: ", loss_mask.detach().cpu().tolist())
    #     print(f"attention_mask shape: {attention_mask.shape}; tokens shape: {tokens.shape}; loss_mask shape: {loss_mask.shape}; labels shape: {labels.shape}")
    #     exit()

    with stimer:
        # print(f"attention_mask shape: {attention_mask.shape}; tokens shape: {tokens.shape}; loss_mask shape: {loss_mask.shape}; labels shape: {labels.shape}")
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, packed_seq_params=packed_seq_params)

    return output_tensor, partial(loss_func, loss_mask, num_seqs)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for SFT ...")

    # train_ds = JSONSFTDataset(args.train_data_path, args.max_padding_length)
    # valid_ds = None
    # if args.valid_data_path:
    #     valid_ds = JSONSFTDataset(args.valid_data_path, args.max_padding_length)
    # test_ds = None
    # if args.test_data_path:
    #     test_ds  = JSONSFTDataset(args.test_data_path, args.max_padding_length)

    train_ds = SFTPreTokenizedDataset(args.train_data_path[0])
    valid_ds = SFTPreTokenizedDataset(args.train_data_path[0]) if args.valid_data_path else None
    test_ds  = SFTPreTokenizedDataset(args.train_data_path[0])  if args.test_data_path  else None

    print_rank_0("> finished creating SFT datasets ...")

    return train_ds, valid_ds, test_ds


def add_extra_args(parser):
    group = parser.add_argument_group(title='extra arguements')

    group.add_argument(
        "--max-padding-length", type=int, default=None, help="max-padding-length"
    )

    return parser


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_extra_args,
        )
