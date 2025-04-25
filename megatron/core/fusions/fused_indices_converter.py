# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl

#from megatron.core.utils import experimental_fn


@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,  # bool
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_idx_offset = row_idx * num_of_local_experts

    # Initialize with 0 and -1 using block-wise operations
    for e in tl.static_range(0, num_of_local_experts, BLOCK_SIZE):
        offset = e + tl.arange(0, BLOCK_SIZE)
        mask = offset < num_of_local_experts
        tl.store(multihot_indices_ptr + row_idx_offset + offset, 0, mask=mask)
        tl.store(probs_in_multihot_ptr + row_idx_offset + offset, 0.0, mask=mask)
        tl.store(position_map_ptr + row_idx_offset + offset, -1, mask=mask)
    tl.debug_barrier()

    # Process each index in the topk list
    for k in tl.static_range(topk):
        index = tl.load(indices_ptr + row_idx * topk + k)
        prob = tl.load(probs_in_indices_ptr + row_idx * topk + k)
        if index != -1 and index < num_of_local_experts:
            tl.store(multihot_indices_ptr + row_idx_offset + index, 1)
            tl.store(probs_in_multihot_ptr + row_idx_offset + index, prob)
            tl.store(position_map_ptr + row_idx_offset + index, k)


@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts

    # Initialize output with zeros
    for k in tl.static_range(topk):
        tl.store(probs_indices_ptr + row_idx * topk + k, 0.0)
    tl.debug_barrier()

    # Process each expert position
    for e in tl.static_range(num_of_local_experts):
        pos = tl.load(position_map_ptr + ptr_offset + e)
        prob = tl.load(probs_in_multihot_ptr + ptr_offset + e)
        if pos != -1 and pos < topk:
            tl.store(probs_indices_ptr + row_idx * topk + pos, prob)


class IndicesToMultihot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        num_of_tokens = indices.shape[0]
        topk = indices.shape[1]
        
        multihot_indices = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda"
        )
        probs_in_multihot = torch.empty_like(multihot_indices, dtype=probs_indices.dtype)
        position_map = torch.full_like(multihot_indices, -1, dtype=torch.int32)

        BLOCK_SIZE = triton.next_power_of_2(max(topk, num_of_local_experts))
        grid = (num_of_tokens,)
        
        _indices_to_multihot_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            num_of_local_experts,
            topk,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4 if BLOCK_SIZE >= 128 else 1,
        )

        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        topk = ctx.topk
        
        grad_probs_indices = torch.zeros(
            (num_of_tokens, topk), 
            dtype=grad_probs_in_multihot.dtype, 
            device="cuda"
        )

        BLOCK_SIZE = triton.next_power_of_2(ctx.num_of_local_experts)
        grid = (num_of_tokens,)
        
        _multihot_to_indices_kernel[grid](
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            ctx.num_of_local_experts,
            topk,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4 if BLOCK_SIZE >= 128 else 1,
        )
        return None, grad_probs_indices, None


#@experimental_fn(introduced_with_version='0.11.0rc0')
def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)