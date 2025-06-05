# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE
import os

try:
    from deep_ep import Buffer

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

FP8_COMM_DEEPEP = os.getenv('FP8_COMM_DEEPEP', '0') == '1' or os.getenv('FP8_COMM_DEEPEP', 'false') == 'true'
if FP8_COMM_DEEPEP:
    try:
        from OpenMixOpl.triton import (
            act_quant_B_ptr as act_quant,
            act_dequant_B_ptr as act_dequant
        )
    except ImportError:
        FP8_COMM_DEEPEP = False

import torch

_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): Input tensor or tuple of fp8 tensors. If a tuple is provided, uses the first tensor in the tuple.

    Returns:
        int: Number of hidden bytes
    """
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)

def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group, previous_event=None, async_finish=False):
        """Forward pass of fused dispatch."""
        # Do Fp8 quantize
        if FP8_COMM_DEEPEP:
            x = act_quant(x)

        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=None,
            async_finish=async_finish,
            allocate_on_comm_stream=False,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.event = event
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        # Do Fp8 dequantize
        if FP8_COMM_DEEPEP and not async_finish:
            recv_x = act_dequant(*recv_x)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, event)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle, previous_event=None
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, grad_token_probs, None, None, None, None

class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, previous_event=None, async_finish=False):
        """Forward pass of fused combine."""
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, event = buffer.combine(
            x, handle=handle, async_finish=async_finish, previous_event=None, allocate_on_comm_stream=False
        )
        ctx.handle = handle
        ctx.group = group

        return combined_x, event

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        # Do Fp8 quantize
        if FP8_COMM_DEEPEP:
            grad_output = act_quant(grad_output)

        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))

        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous() if isinstance(grad_output, torch.Tensor) else grad_output,
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Do Fp8 dequantize
        if FP8_COMM_DEEPEP:
            grad_x = act_dequant(*grad_x)

        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(x, token_indices, token_probs, num_experts, group, previous_event=None, async_finish=False):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(x.contiguous(), token_indices, token_probs, num_experts, group, previous_event, async_finish)

    def wait_dispatch_finish(hidden_states, dispatch_event):
        dispatch_event.current_stream_wait()
        if FP8_COMM_DEEPEP:
            hidden_states = act_dequant(*hidden_states)
        return hidden_states

    def fused_combine(x, group, handle, previous_event=None, async_finish=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, previous_event, async_finish)

    def wait_combine_finish(hidden_states, combine_event):
        combine_event.current_stream_wait()
        return hidden_states

else:
    fused_dispatch = None
    fused_combine = None
    wait_dispatch_finish = None
    wait_combine_finish megatron/core/transformer/moe/fused_a2a.py= None