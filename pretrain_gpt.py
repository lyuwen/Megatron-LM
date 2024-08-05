# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial

from typing import Union
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
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


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
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

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
            rotary_base=args.rotary_base
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
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
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
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
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


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
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path = args.s3_cache_path
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def extra_valid_core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    extra_valid_configs = []

    for datalist in args.extra_valid_datalist:
        with open(datalist, 'rt') as f:
          datalist_input = f.read().strip().split()
          extra_valid_configs.append(GPTDatasetConfig(
              random_seed=args.seed,
              sequence_length=args.seq_length,
              #  blend=None,
              blend_per_split=[
                  None,
                  get_blend_from_list(datalist_input),
                  None,
              ],
              #  split=None,
              num_dataset_builder_threads=args.num_dataset_builder_threads,
              path_to_cache=args.data_cache_path,
              mmap_bin_files=args.mmap_bin_files,
              tokenizer=tokenizer,
              reset_position_ids=args.reset_position_ids,
              reset_attention_mask=args.reset_attention_mask,
              eod_mask_loss=args.eod_mask_loss,
              create_attention_mask=args.create_attention_mask_in_dataloader,
          ))
    return extra_valid_configs


def extra_valid_datasets_provider(extra_valid_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    configs = extra_valid_core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    valid_ds_list = []

    for config, num_samples in zip(configs, extra_valid_num_samples):
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            (None, num_samples, None),
            is_dataset_built_on_rank,
            config
        ).build()
        valid_ds_list.append(valid_ds)

    print_rank_0("> finished creating GPT datasets for extra validation sets ...")

    return valid_ds_list


def add_extra_args(parser):
    group = parser.add_argument_group(title='extra arguements')

    group.add_argument('--extra-valid-datalist', type=str, default=None, action="append",
                       help='A list of dataset lists containing additional validation datasets. '
                       )
    group.add_argument('--extra-valid-data-samples', type=int, default=None, action="append",
                       help='Sample sizes of the list of dataset lists containing additional validation datasets. '
                           'The last incomplete batch will be droped, but will always up-sample to at least 1 global batch.'
                       )
    group.add_argument('--extra-valid-data-names', type=str, default=None, action="append",
                       help='Names of the dataset lists containing additional validation datasets. '
                       )

    return parser


def build_extra_valid_data_loaders(
        build_extra_valid_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    print_rank_0('> building dataloaders for extra validation datasets ...')

    valid_dataloaders, extra_valid_data_samples, extra_valid_data_names = (None, None, None)

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_extra_valid_datasets_provider, "is_distributed", False)

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

        assert len(args.extra_valid_data_samples) == len(args.extra_valid_datalist), \
            "Length of the datalist and sample sizes do not match."
        # At least up-sample to 1 global batch
        extra_valid_data_samples = [max(num_samples // args.global_batch_size, 1) * args.global_batch_size for num_samples in args.extra_valid_data_samples]
        args.extra_valid_data_samples = extra_valid_data_samples
        #
        if args.extra_valid_data_names:
            assert len(args.extra_valid_data_names) == len(args.extra_valid_datalist), \
                "Length of the datalist and data names do not match."
            extra_valid_data_names = args.extra_valid_data_names
        else:
            extra_valid_data_names = [f"Extra data {i}" for i in range(len(args.extra_valid_datalist))]
        args.extra_valid_data_names = extra_valid_data_names
        # Build datasets.
        valid_ds_list = build_extra_valid_datasets_provider(extra_valid_data_samples)
        # Build dataloders.
        orig_dataloader_type = args.dataloader_type
        args.dataloader_type = "cyclic"
        valid_dataloaders = [build_pretraining_data_loader(valid_ds, 0) for valid_ds in valid_ds_list]
        args.dataloader_type = orig_dataloader_type

    print_rank_0('> finished building dataloaders for extra validation datasets ...')

    return valid_dataloaders, extra_valid_data_samples, extra_valid_data_names


def build_extra_valid_data_iterators(extra_valid_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    #  valid_dataloaders = build_extra_valid_data_loaders(extra_valid_datasets_provider)
    valid_dataloaders, extra_valid_data_samples, extra_valid_data_names = \
        build_extra_valid_data_loaders(extra_valid_datasets_provider)

    # Build iterators.
    def cyclic_iter(iter):
        while True:
            for x in iter:
                yield x

    if valid_dataloaders is not None:
        valid_data_iterators = [iter(cyclic_iter(dataloader)) for dataloader in valid_dataloaders]
    else:
        valid_data_iterators = None

    return valid_data_iterators, extra_valid_data_samples, extra_valid_data_names


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True
    extra_valid_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_extra_args,
        extra_valid_data_iterators_builder=partial(build_extra_valid_data_iterators, extra_valid_datasets_provider=extra_valid_datasets_provider),
        )
