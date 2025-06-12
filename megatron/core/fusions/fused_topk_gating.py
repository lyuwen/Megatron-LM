# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl
import os


@triton.jit
def fused_topk_gating_forward_kernel_with_group(
    # temp
    scores_temp_ptr,
    group_view_temp_ptr,
    top_values_temp_ptr,
    top_indices_temp2_ptr,
    group_mask_temp_ptr,
    # input
    logits_ptr,
    expert_bias_ptr,
    has_expert_bias: tl.constexpr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # params
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    num_groups: tl.constexpr,
    experts_per_group: tl.constexpr,
    topk: tl.constexpr,
    group_topk: tl.constexpr,
    topk_per_group: tl.constexpr,
    scaling_factor: tl.constexpr,
    score_function: tl.constexpr,
    # block size
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
    BLOCK_SIZE_GROUP_TOPK: tl.constexpr,
    BLOCK_SIZE_TOPK_PER_GROUP: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    group_offs = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)

    group_topk_offs = tl.arange(0, BLOCK_SIZE_GROUP_TOPK)
    group_topk_mask = group_topk_offs < group_topk

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load input
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    if score_function == "softmax":
        logits_fp32 = logits.to(tl.float32)
        scores_fp32 = tl.softmax(logits_fp32)
        scores = scores_fp32.to(logits.dtype)
        # scores = tl.softmax(logits)
        scores_for_routing = scores

        tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)

        # ##########################
        # ## compute group_scores ##
        # ##########################
        
        tl.store(group_view_temp_ptr + pid * num_experts + expert_offs, scores_for_routing, mask=expert_mask)
        
        tl.debug_barrier()

        # group_view = tl.reshape(scores_for_routing, num_groups, experts_per_group)

        offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]
        offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]
        indices = offs_m * experts_per_group + offs_n
        mask = (offs_m < num_groups) & (offs_n < experts_per_group)
        group_view = tl.load(group_view_temp_ptr + pid * num_experts + indices, mask=mask)

        # topk_2D   
        sorted_group_view = tl.sort(group_view, descending=True)
        tl.store(group_view_temp_ptr + pid * num_groups * experts_per_group + indices, sorted_group_view, mask=mask)

        tl.debug_barrier()

        offs_n2 = tl.arange(0, BLOCK_SIZE_TOPK_PER_GROUP)[None, :]
        top_values_offs = pid * num_groups * experts_per_group + offs_m * experts_per_group + offs_n2
        top_values_mask = (offs_m < num_groups) & (offs_n2 < topk_per_group)
        top_values = tl.load(top_values_temp_ptr + top_values_offs, mask=top_values_mask)

        group_scores = tl.sum(top_values, axis=-1)

        # #######################
        # ## compute group_idx ##
        # #######################

        data = group_scores

        for i in range(group_topk):
            max_idx = tl.argmax(data, axis=0)
            tl.store(top_indices_temp2_ptr + pid * group_topk + i, max_idx)
            data = tl.where(group_offs == max_idx, -float('inf'), data)

        tl.debug_barrier()

        group_idx = tl.load(top_indices_temp2_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
        
        # ########################
        # ## compute group_mask ##
        # ########################

        ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)
        tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

        tl.debug_barrier()

        # ########################
        # ## compute score_mask ##
        # ########################

        expert_group_idx = expert_offs // experts_per_group
        score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

        # ###########################
        # ## compute masked_scores ##
        # ###########################

        score_mask_bool = score_mask != 0
        masked_scores = tl.where(score_mask_bool, scores_for_routing, -float('inf'))

        # #########################
        # ## compute top_indices ##
        # #########################

        data = masked_scores

        for i in range(topk):
            max_idx = tl.argmax(data, axis=0)
            tl.store(top_indices_ptr + pid * topk + i, max_idx)
            data = tl.where(expert_offs == max_idx, -float('inf'), data)

        tl.debug_barrier()

        top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)
        probs = tl.load(scores_temp_ptr + pid * num_experts + top_indices, mask=topk_mask)

        tl.store(top_scores_ptr + pid * topk + topk_offs, probs, mask=topk_mask)

    elif score_function == "sigmoid":
        logits_fp32 = logits.to(tl.float32)
        logits_fp32 = tl.sigmoid(logits_fp32)
        if has_expert_bias:
            expert_bias = tl.load(expert_bias_ptr + expert_offs, mask=expert_mask)
            logits_fp32 += expert_bias
        tl.store(scores_temp_ptr + pid * num_experts + expert_offs, logits_fp32, mask=expert_mask)
        
        # ##########################
        # ## compute group_scores ##
        # ##########################
        # group_view = tl.reshape(scores_for_routing, num_groups, experts_per_group)
        
        tl.store(group_view_temp_ptr + pid * num_experts + expert_offs, logits_fp32, mask=expert_mask)
        
        tl.debug_barrier()

        offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]
        offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]
        indices = offs_m * experts_per_group + offs_n
        mask = (offs_m < num_groups) & (offs_n < experts_per_group)
        group_view = tl.load(group_view_temp_ptr + pid * num_experts + indices, mask=mask)

        # topk_2D
        sorted_group_view = tl.sort(group_view, descending=True)
        tl.store(group_view_temp_ptr + pid * num_groups * experts_per_group + indices, sorted_group_view, mask=mask)

        tl.debug_barrier()

        offs_n2 = tl.arange(0, BLOCK_SIZE_TOPK_PER_GROUP)[None, :]
        top_values_offs = pid * num_groups * experts_per_group + offs_m * experts_per_group + offs_n2
        top_values_mask = (offs_m < num_groups) & (offs_n2 < topk_per_group)
        top_values = tl.load(top_values_temp_ptr + top_values_offs, mask=top_values_mask)

        group_scores = tl.sum(top_values, axis=-1)

        # #######################
        # ## compute group_idx ##
        # #######################

        data = group_scores

        for i in range(group_topk):
            max_idx = tl.argmax(data, axis=0)
            tl.store(top_indices_temp2_ptr + pid * group_topk + i, max_idx)
            data = tl.where(group_offs == max_idx, -float('inf'), data)

        tl.debug_barrier()

        group_idx = tl.load(top_indices_temp2_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
        
        # ########################
        # ## compute group_mask ##
        # ########################

        ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)
        tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

        tl.debug_barrier()

        # ########################
        # ## compute score_mask ##
        # ########################

        expert_group_idx = expert_offs // experts_per_group
        score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

        # ###########################
        # ## compute masked_scores ##
        # ###########################

        score_mask_bool = score_mask != 0
        masked_scores = tl.where(score_mask_bool, logits_fp32, -float('inf'))

        # #########################
        # ## compute top_indices ##
        # #########################

        data = masked_scores

        for i in range(topk):
            max_idx = tl.argmax(data, axis=0)
            tl.store(top_indices_ptr + pid * topk + i, max_idx)
            data = tl.where(expert_offs == max_idx, -float('inf'), data)

        tl.debug_barrier()

        top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)
        top_scores = tl.load(scores_temp_ptr + pid * num_experts + top_indices, mask=topk_mask)

        tl.store(top_scores_ptr + pid * topk + topk_offs, top_scores, mask=topk_mask)

        # ###################
        # ## compute probs ##
        # ###################

        if topk > 1:
            sum_top_scores = tl.sum(top_scores, axis=0)
            probs = top_scores / (sum_top_scores + 1e-20)
        else:
            probs = top_scores

    else:
        raise ValueError(f"Invalid score function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    
    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def fused_topk_gating_forward_kernel_without_group(
    # temp
    scores_temp_ptr,
    # input
    logits_ptr,
    expert_bias_ptr,
    has_expert_bias: tl.constexpr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # params
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    score_function: tl.constexpr,
    # block size
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load input
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    if score_function == "softmax":
        logits_fp32 = logits.to(tl.float32)
        scores_fp32 = tl.softmax(logits_fp32)
        scores = scores_fp32.to(logits.dtype)

        tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)

        # topk logits (num_experts -> topk)
        data = scores
        for i in range(topk):
            max_val = tl.max(data, axis=0)
            max_idx = tl.argmax(data, axis=0)
            tl.store(top_scores_ptr + pid * topk + i, max_val)
            tl.store(top_indices_ptr + pid * topk + i, max_idx)
            data = tl.where(expert_offs == max_idx, -float('inf'), data)

        tl.debug_barrier()
        probs = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
        top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    elif score_function == "sigmoid":
        logits_fp32 = logits.to(tl.float32)
        logits_fp32 = tl.sigmoid(logits_fp32)
        if has_expert_bias:
            expert_bias = tl.load(expert_bias_ptr + expert_offs, mask=expert_mask)
            logits_fp32 += expert_bias
        tl.store(scores_temp_ptr + pid * num_experts + expert_offs, logits_fp32, mask=expert_mask)

        # topk logits (num_experts -> topk)
        data = logits_fp32
        for i in range(topk):
            max_val = tl.max(data, axis=0)
            max_idx = tl.argmax(data, axis=0)
            tl.store(top_scores_ptr + pid * topk + i, max_val)
            tl.store(top_indices_ptr + pid * topk + i, max_idx)
            data = tl.where(expert_offs == max_idx, -float('inf'), data)

        tl.debug_barrier()
        top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
        top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)
        scores = top_scores.to(logits.dtype)

        # compute probs
        if topk > 1:
            sum_scores = tl.sum(scores) + 1e-20
            probs = scores / sum_scores
        else:
            probs = scores
    else:
        raise ValueError(f"Invalid score function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    
    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def triton_backward_kernel_with_group(
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
    scaling_factor: tl.constexpr,
    score_function: tl.constexpr,
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

    # scaling_factor grad_probs
    if scaling_factor:
        grad_probs = grad_probs * scaling_factor

    # compute grad_scores
    if score_function == "softmax":
        grad_scores = grad_probs
        
        # compute grad_logits(1)
        tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_scores, mask=topk_mask)

        # compute softmax
        logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
        logits_fp32 = logits.to(tl.float32)
        scores_fp32 = tl.softmax(logits_fp32)
        scores = scores_fp32.to(logits.dtype)
        
        # compute grad_logits(2)
        tl.debug_barrier()
        grad_logits = tl.load(grad_logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
        sum_grad = tl.sum(scores * grad_logits, axis=0)
        grad_logits = scores * (grad_logits - sum_grad)
    elif score_function == "sigmoid":
        if topk > 1:
            top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
            sum_top_scores = tl.sum(top_scores, -1) + 1e-20
            grad_scores = (grad_probs * sum_top_scores - tl.sum((top_scores * grad_probs), -1)) / (sum_top_scores * sum_top_scores)
        else:
            grad_scores = grad_probs

        # compute grad_logits(1)
        tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_scores, mask=topk_mask)

        # compute sig
        logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
        logits_fp32 = logits.to(tl.float32)
        sig = tl.sigmoid(logits_fp32)

        # compute grad_logits(2)
        tl.debug_barrier()
        grad_logits = tl.load(grad_logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
        grad_logits *= sig * (1 - sig)
    tl.store(grad_logits_ptr + pid * num_experts + expert_offs, grad_logits, mask=expert_mask)

@triton.jit
def triton_backward_kernel_without_group(
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
    scaling_factor: tl.constexpr,
    score_function: tl.constexpr,
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

    # scaling_factor grad_probs
    if scaling_factor:
        grad_probs = grad_probs * scaling_factor

    # compute grad_scores
    if score_function == "softmax":
        grad_scores = grad_probs
        
        # compute grad_logits(1)
        tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_scores, mask=topk_mask)

        # compute softmax
        logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
        logits_fp32 = logits.to(tl.float32)
        scores_fp32 = tl.softmax(logits_fp32)
        scores = scores_fp32.to(logits.dtype)
        
        # compute grad_logits(2)
        tl.debug_barrier()
        grad_logits = tl.load(grad_logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
        sum_grad = tl.sum(scores * grad_logits, axis=0)
        grad_logits = scores * (grad_logits - sum_grad)
    elif score_function == "sigmoid":
        if topk > 1:
            top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
            sum_top_scores = tl.sum(top_scores, -1) + 1e-20
            grad_scores = (grad_probs * sum_top_scores - tl.sum((top_scores * grad_probs), -1)) / (sum_top_scores * sum_top_scores)
        else:
            grad_scores = grad_probs

        # compute grad_logits(1)
        tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_scores, mask=topk_mask)

        # compute sig
        logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
        logits_fp32 = logits.to(tl.float32)
        sig = tl.sigmoid(logits_fp32)

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
        capacity_factor,
        pad_to_capacity,
        drop_policy,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        deterministic_mode,
        score_function,
        expert_bias,
    ):
        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
        assert capacity_factor is None, "capacity_factor must be None"
        assert use_pre_softmax is True, "use_pre_softmax must be True to ensure the same computation process"
        
        # params
        num_tokens, num_experts = logits.shape
        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        # output
        topk_masked_gates = torch.zeros_like(logits)
        topk_map = torch.zeros_like(logits).bool()

        top_indices = torch.empty([num_tokens, topk], device=logits.device, dtype=torch.int64)
        top_scores = torch.empty([num_tokens, topk], device=logits.device, dtype=logits.dtype)

        # temp values
        scores_temp = torch.empty([num_tokens, num_experts], device=logits.device, dtype=logits.dtype)

        grid = lambda meta: (num_tokens,)
        if group_topk:
            # 分组处理相关的参数计算
            experts_per_group = num_experts // num_groups
            topk_per_group = topk // group_topk
            
            # 为分组处理创建额外的临时变量
            BLOCK_SIZE_NUM_GROUPS = triton.next_power_of_2(num_groups)
            BLOCK_SIZE_EXPERTS_PER_GROUP = triton.next_power_of_2(experts_per_group)
            BLOCK_SIZE_GROUP_TOPK = triton.next_power_of_2(group_topk)
            BLOCK_SIZE_TOPK_PER_GROUP = triton.next_power_of_2(topk_per_group)

            group_view_temp = torch.empty([num_tokens, num_groups, experts_per_group], device=logits.device, dtype=logits.dtype)
            top_values_temp = torch.empty([num_tokens, num_groups, topk_per_group], device=logits.device, dtype=logits.dtype)
            top_indices_temp2 = torch.empty([num_tokens, group_topk], device=logits.device, dtype=torch.int64)
            group_mask_temp = torch.zeros([num_tokens, num_groups], device=logits.device, dtype=logits.dtype)

            print(group_topk)
            fused_topk_gating_forward_kernel_with_group[grid](
                # temp
                scores_temp,
                group_view_temp,
                top_values_temp,
                top_indices_temp2,
                group_mask_temp,
                # input
                logits,
                expert_bias if expert_bias is not None else torch.empty(0, device=logits.device),
                expert_bias is not None,
                # output
                topk_masked_gates,
                topk_map,
                top_indices,
                top_scores,
                # params
                num_tokens,
                num_experts,
                num_groups,
                experts_per_group,
                topk,
                group_topk,
                topk_per_group,
                scaling_factor,
                score_function,
                # block size
                BLOCK_SIZE_NUM_EXPERTS,
                BLOCK_SIZE_NUM_GROUPS,
                BLOCK_SIZE_EXPERTS_PER_GROUP,
                BLOCK_SIZE_TOPK,
                BLOCK_SIZE_GROUP_TOPK,
                BLOCK_SIZE_TOPK_PER_GROUP
            )
            print(group_topk)
        else:
            fused_topk_gating_forward_kernel_without_group[grid](
                # temp
                scores_temp,
                # input
                logits,
                expert_bias if expert_bias is not None else torch.empty(0, device=logits.device),
                expert_bias is not None,
                # output
                topk_masked_gates,
                topk_map,
                top_indices,
                top_scores,
                # params
                num_tokens,
                num_experts,
                topk,
                scaling_factor,
                score_function,
                # block size
                BLOCK_SIZE_NUM_EXPERTS,
                BLOCK_SIZE_TOPK
            )

        ctx.save_for_backward(logits, top_indices, top_scores)
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.topk = topk
        ctx.num_groups = num_groups
        ctx.group_topk = group_topk
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function

        tokens_per_expert = topk_map.sum(dim=0)
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
        scaling_factor = ctx.scaling_factor
        score_function = ctx.score_function
        group_topk = ctx.group_topk

        # params
        num_tokens, num_experts = logits.shape
        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        # output
        grad_logits = torch.zeros_like(logits)
        
        grid = lambda meta: (num_tokens,)
        if group_topk:
            triton_backward_kernel_with_group[grid](
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
                scaling_factor,
                score_function,
                BLOCK_SIZE_NUM_EXPERTS,
                BLOCK_SIZE_TOPK
            )
        else:
            triton_backward_kernel_without_group[grid](
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
                scaling_factor,
                score_function,
                BLOCK_SIZE_NUM_EXPERTS,
                BLOCK_SIZE_TOPK
            )
        return grad_logits, None, None, None, None, None, None, None, None, None, None, None


def fused_topk_gating_without_capacity(
                    logits, 
                    topk,
                    capacity_factor,
                    pad_to_capacity,
                    drop_policy,
                    use_pre_softmax,
                    num_groups,
                    group_topk,
                    scaling_factor,
                    deterministic_mode,
                    score_function,
                    expert_bias,
                    ):
    return TopkGating.apply(
                            logits,
                            topk,
                            capacity_factor,
                            pad_to_capacity,
                            drop_policy,
                            use_pre_softmax,
                            num_groups,
                            group_topk,
                            scaling_factor,
                            deterministic_mode,
                            score_function,
                            expert_bias,
                            )