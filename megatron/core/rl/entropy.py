# Copyright 2025 ZJLab
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for using tensor_parallel in megatron
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.nn import init
from collections import deque


_ENTROPY_STORE = []


def push_micro_batch(item):
    global = _ENTROPY_STORE
    _ENTROPY_STORE.append(_ENTROPY_STORE)


def get_batch():
    global _ENTROPY_STORE
    if not _ENTROPY_STORE:
        return None
    batch = torch.cat(_ENTROPY_STORE)
    _ENTROPY_STORE = []
    return batch


def get_batch_entropy():
    seq_entropy = get_batch()
    if not seq_entropy:
        return None
    entropy = torch.mean(seq_entropy)  # seq-mean
    return entropy


try:
    from megatron.core import parallel_state as mpu
    class _VocabParallelEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
            @torch.compile(dynamic=True)
            def mul_reduce(a, b):
                return (a * b).sum(dim=-1, keepdim=True)

            logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
            normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
            normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
            normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
            dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
            softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
            sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
            dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
            entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
            ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
            return entropy.squeeze(dim=-1)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
            # reuse softmax_logits as grad
            vocab_parallel_logits.sub_(sum_softmax_times_logits)
            softmax_logits.mul_(vocab_parallel_logits)
            softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
            # recover vocab_parallel_logits
            vocab_parallel_logits.add_(sum_softmax_times_logits)
            softmax_logits.mul_(-1)
            return softmax_logits


    def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy when the logits are sharded in tp ranks

        Args:
            vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

        Returns: (total_nnz,)

        """
        return _VocabParallelEntropy.apply(vocab_parallel_logits)
except ImportError:
    pass


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss
