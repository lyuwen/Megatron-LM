# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def triton_forward_kernel(
    # input
    logits_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # params
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load logits_fp32
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask).to(tl.float32)
    scores = tl.sigmoid(logits)

    # topk logits (num_experts -> topk)
    data = scores
    for i in range(topk):
        max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_scores_ptr + pid * topk + i, max_val)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)

    tl.debug_barrier()

    # load topk output
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute probs
    if topk > 1:
        sum_top_scores = tl.sum(top_scores) + 1e-20
        probs = top_scores / sum_top_scores
    else:
        probs = top_scores

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    
    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def triton_backward_kernel(
    # input
    logits_ptr,
    top_indices_ptr,
    top_scores_ptr,
    grad_topk_masked_gates_ptr,
    # output
    grad_logits_ptr,
    # params
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load top_indices
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute grad_probs
    grad_probs = tl.load(grad_topk_masked_gates_ptr + pid * num_experts + top_indices, mask=topk_mask)

    # compute grad_scores
    if topk > 1:
        top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
        sum_top_scores = tl.sum(top_scores)
        grad_scores = (grad_probs * sum_top_scores - tl.sum(top_scores * grad_probs)) / (sum_top_scores * sum_top_scores)
    else:
        grad_scores = grad_probs

    # compute grad_logits(1)
    tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_scores, mask=topk_mask)

    # compute sig
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask).to(tl.float32)
    sig = tl.sigmoid(logits)

    # compute grad_logits(2)
    tl.debug_barrier()
    grad_logits = tl.load(grad_logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    grad_logits *= sig * (1 - sig)
    tl.store(grad_logits_ptr + pid * num_experts + expert_offs, grad_logits, mask=expert_mask)


class TopkGating(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        topk,
    ):
        ctx.topk = topk
        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
        # params
        num_tokens, num_experts = logits.shape

        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        # output
        topk_masked_gates = torch.zeros_like(logits)
        topk_map = torch.zeros_like(logits).bool()

        top_indices = torch.empty([num_tokens, topk], device=logits.device, dtype=torch.int64)
        top_scores = torch.empty([num_tokens, topk], device=logits.device, dtype=logits.dtype)

        grid = lambda meta: (num_tokens,)
        triton_forward_kernel[grid](
            # input
            logits,
            # output
            topk_masked_gates,
            topk_map,
            top_indices,
            top_scores,
            # params
            num_experts,
            topk,
            BLOCK_SIZE_NUM_EXPERTS,
            BLOCK_SIZE_TOPK
        )
        tokens_per_expert = topk_map.sum(dim=0)
        ctx.save_for_backward(logits, top_indices, top_scores)
        return topk_masked_gates, topk_map, tokens_per_expert

    @staticmethod
    def backward(
        ctx,
        grad_topk_masked_gates,
        grad_topk_map,
        grad_tokens_per_expert
    ):
        logits, top_indices, top_scores = ctx.saved_tensors
        topk = ctx.topk

        # params
        num_tokens, num_experts = logits.shape
        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        # output
        grad_logits = torch.zeros_like(logits)
        
        grid = lambda meta: (num_tokens,)
        triton_backward_kernel[grid](
            # input
            logits,
            top_indices,
            top_scores,
            grad_topk_masked_gates,
            # output
            grad_logits,
            # params
            num_experts,
            topk,
            BLOCK_SIZE_NUM_EXPERTS,
            BLOCK_SIZE_TOPK
        )
        return grad_logits, None


def fused_topk_gating(logits, topk):
    return TopkGating.apply(logits, topk)