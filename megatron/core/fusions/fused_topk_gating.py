# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl

#from megatron.core.utils import experimental_fn


@triton.jit
def fused_topk_gating_forward_kernel(
    # temp
    scores_temp_ptr,
    top_values_temp_ptr,
    top_indices_temp2_ptr,
    group_mask_temp_ptr,
    # input
    logits_ptr,
    expert_bias_ptr,
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
    # expert_bias = tl.load(expert_bias_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    expert_bias = tl.load(expert_bias_ptr + expert_offs, mask=expert_mask)

    # tl.debug_barrier()

    # ####################
    # ## compute scores ##
    # ####################

    scores = tl.sigmoid(logits)
    scores_for_routing = scores + expert_bias

    tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)

    # ##########################
    # ## compute group_scores ##
    # ##########################

    group_view = tl.reshape(scores_for_routing, num_groups, experts_per_group)

    # topk_2D
    data = group_view # 其实不用保留group_view

    offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]         # 组内行偏移 num_groups
    offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]  # 组内列偏移 experts_per_group

    for i in range(topk_per_group):
        max_idx = tl.argmax(data, axis=1)  # 形状: (BLOCK_M,)
        max_val = tl.max(data, axis=1)     # 形状: (BLOCK_M,)
        # 扩展为二维 (BLOCK_M, 1)，与输出地址对齐
        max_idx_reshaped = max_idx[:, None]
        max_val_reshaped = max_val[:, None]
        # 计算输出地址：行优先，每行 K 个值
        index = pid * num_groups * topk_per_group + offs_m * topk_per_group + i  # 形状: (BLOCK_M, 1)
        # 存储结果（无需显式掩码，因 data 已经过滤无效值）
        tl.store(top_values_temp_ptr + index, max_val_reshaped)
        # tl.store(top_indices_temp_ptr + index, max_idx_reshaped)
        # 屏蔽已选元素（避免重复选取）
        data_mask = (offs_n == max_idx[:, None])  # 广播匹配每行的列索引
        data = tl.where(data_mask, -float('inf'), data)

    tl.debug_barrier()

    # load topk_2D output
    offs_n2 = tl.arange(0, BLOCK_SIZE_TOPK_PER_GROUP)[None, :] # 组内列偏移 topk_per_group

    top_values_offs = pid * num_groups * topk_per_group + offs_m * topk_per_group + offs_n2
    top_values_mask = (offs_m < num_groups) & (offs_n2 < topk_per_group)
    top_values = tl.load(top_values_temp_ptr + top_values_offs, mask=top_values_mask)

    group_scores = tl.sum(top_values, axis=-1)

    # #######################
    # ## compute group_idx ##
    # #######################

    # topk_1D
    data = group_scores

    for i in range(group_topk):
        # max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        # tl.store(top_values_temp2_ptr + pid * group_topk + i, max_val)
        tl.store(top_indices_temp2_ptr + pid * group_topk + i, max_idx)
        data = tl.where(group_offs == max_idx, -float('inf'), data)

    tl.debug_barrier()

    # load topk_1D output
    group_idx = tl.load(top_indices_temp2_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
    
    # ########################
    # ## compute group_mask ##
    # ########################

    ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)

    tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

    # 需要同步以解决score_mask错误问题
    tl.debug_barrier()

    # 这里一整段可不可以变成利用group_idx进行load来置1, 其他位置用others=0来置0?

    # ########################
    # ## compute score_mask ##
    # ########################

    expert_group_idx = expert_offs // experts_per_group # 构成专家与其组id映射

    score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

    # ###########################
    # ## compute masked_scores ##
    # ###########################

    # warning: score_mask is not bool
    # masked_scores = tl.where(score_mask, scores_for_routing, -float('inf'))
    score_mask_bool = score_mask != 0
    masked_scores = tl.where(score_mask_bool, scores_for_routing, -float('inf'))

    # #########################
    # ## compute top_indices ##
    # #########################

    # topk_1D
    data = masked_scores

    for i in range(topk):
        # max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        # tl.store(top_values_temp3_ptr + pid * topk + i, max_val)
        # tl.store(top_indices_temp3_ptr + pid * topk + i, max_idx)

        # saved for backward
        tl.store(top_indices_ptr + pid * topk + i, max_idx)

        data = tl.where(expert_offs == max_idx, -float('inf'), data)

    tl.debug_barrier()

    # load topk_1D output
    # top_indices = tl.load(top_indices_temp3_ptr + pid * topk + topk_offs, mask=topk_mask)

    # saved for backward(top_indices_ptr)
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # Not continuous store+load of a common ptr. Need?
    # tl.debug_barrier()

    # ########################
    # ## compute top_scores ##
    # ########################

    top_scores = tl.load(scores_temp_ptr + pid * num_experts + top_indices, mask=topk_mask)

    # saved for backward
    tl.store(top_scores_ptr + pid * topk + topk_offs, top_scores, mask=topk_mask)

    # ###################
    # ## compute probs ##
    # ###################

    sum_top_scores = tl.sum(top_scores, axis=0)
    probs = top_scores / (sum_top_scores + 1e-20)

    # ###############################
    # ## compute topk_masked_gates ##
    # ###############################

    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)

    # ######################
    # ## compute topk_map ##
    # ######################

    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)

    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)


@triton.jit
def fused_topk_gating_backward_kernel(
    # input
    logits_ptr,
    top_indices_ptr,
    top_scores_ptr,
    grad_topk_masked_gates_ptr,
    # output
    grad_logits_ptr,
    # params
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load input
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)

    # ########################
    # ## compute grad_probs ##
    # ########################

    grad_probs = tl.load(grad_topk_masked_gates_ptr + pid * num_experts + top_indices, mask=topk_mask)

    # ############################
    # ## compute sum_top_scores ##
    # ############################

    sum_top_scores = tl.sum(top_scores, -1) + 1e-20

    # #############################
    # ## compute grad_top_scores ##
    # #############################

    grad_top_scores = (grad_probs * sum_top_scores - tl.sum((top_scores * grad_probs), -1)) / (sum_top_scores * sum_top_scores)

    # #########################
    # ## compute grad_logits ##
    # #########################
        
    data = grad_top_scores * top_scores * (1 - top_scores)
    tl.store(grad_logits_ptr + pid * num_experts + top_indices, data, mask=topk_mask)


class TopkGating(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        topk,
        num_groups,
        group_topk,
        expert_bias,
    ):
        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
        # params
        num_tokens, num_experts = logits.shape
        experts_per_group = num_experts // num_groups
        topk_per_group = topk // group_topk

        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_NUM_GROUPS = triton.next_power_of_2(num_groups)
        BLOCK_SIZE_EXPERTS_PER_GROUP = triton.next_power_of_2(experts_per_group)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)
        BLOCK_SIZE_GROUP_TOPK = triton.next_power_of_2(group_topk)
        BLOCK_SIZE_TOPK_PER_GROUP = triton.next_power_of_2(topk_per_group)

        # output
        topk_masked_gates = torch.zeros_like(logits)
        topk_map = torch.zeros_like(logits).bool()

        # top_indices = torch.empty([num_tokens, topk], device=logits.device, dtype=logits.dtype)
        top_indices = torch.empty([num_tokens, topk], device=logits.device, dtype=torch.int64)
        top_scores = torch.empty([num_tokens, topk], device=logits.device, dtype=logits.dtype)

        # temp values
        scores_temp = torch.empty([num_tokens, num_experts], device=logits.device, dtype=logits.dtype)
        top_values_temp = torch.empty([num_tokens, num_groups, topk_per_group], device=logits.device, dtype=logits.dtype)
        top_indices_temp2 = torch.empty([num_tokens, group_topk], device=logits.device, dtype=torch.int64)
        group_mask_temp = torch.zeros([num_tokens, num_groups], device=logits.device, dtype=logits.dtype)

        grid = lambda meta: (num_tokens,)
        fused_topk_gating_forward_kernel[grid](
            # temp
            scores_temp,
            top_values_temp,
            top_indices_temp2,
            group_mask_temp,
            # input
            logits,
            expert_bias,
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
            BLOCK_SIZE_NUM_EXPERTS,
            BLOCK_SIZE_NUM_GROUPS,
            BLOCK_SIZE_EXPERTS_PER_GROUP,
            BLOCK_SIZE_TOPK,
            BLOCK_SIZE_GROUP_TOPK,
            BLOCK_SIZE_TOPK_PER_GROUP
        )
        ctx.save_for_backward(logits, top_indices, top_scores)
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.topk = topk

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
        num_tokens = ctx.num_tokens
        num_experts = ctx.num_experts
        topk = ctx.topk

        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        grad_logits = torch.zeros([num_tokens, num_experts], device=logits.device, dtype=logits.dtype)

        grid = lambda meta: (num_tokens,)
        fused_topk_gating_backward_kernel[grid](
            # input
            logits,
            top_indices,
            top_scores,
            grad_topk_masked_gates,
            # output
            grad_logits,
            # params
            num_tokens,
            num_experts,
            topk,
            BLOCK_SIZE_NUM_EXPERTS,
            BLOCK_SIZE_TOPK
        )
        return grad_logits, None, None, None, None


def fused_topk_gating(logits, topk, num_groups, group_topk, expert_bias):
    return TopkGating.apply(logits, topk, num_groups, group_topk, expert_bias)