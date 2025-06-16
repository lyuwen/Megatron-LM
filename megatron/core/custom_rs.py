# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import functools
from typing import Callable

class RecomputeController:
    """重计算控制器，管理全局的重计算策略"""
    
    def __init__(self):
        # 全局开关
        self.global_enabled = os.getenv('ENABLE_CUSTOM_RECOMPUTE', 'false').lower() == 'true'
        
        # 环境变量控制开关
        self.component_switches = {
            'moe': os.getenv('RECOMPUTE_MOE', 'true').lower() == 'true',
            'mlp': os.getenv('RECOMPUTE_MLP', 'true').lower() == 'true',
            'attn': os.getenv('RECOMPUTE_ATTN', 'true').lower() == 'true',

            'router': os.getenv('RECOMPUTE_ROUTER', 'true').lower() == 'true',
            'permutation': os.getenv('RECOMPUTE_PERMUTATION', 'true').lower() == 'true',
            'experts': os.getenv('RECOMPUTE_EXPERTS', 'true').lower() == 'true',
            'unpermutation': os.getenv('RECOMPUTE_UNPERMUTATION', 'true').lower() == 'true',
            'shared_experts': os.getenv('RECOMPUTE_SHARED_EXPERTS', 'true').lower() == 'true',
            
            'expert_fc1': os.getenv('RECOMPUTE_EXPERT_FC1', 'true').lower() == 'true',
            'expert_bias_act': os.getenv('RECOMPUTE_EXPERT_BIAS_ACT', 'true').lower() == 'true',
            'expert_fc2': os.getenv('RECOMPUTE_EXPERT_FC2', 'true').lower() == 'true',

            'attn_core': os.getenv('RECOMPUTE_ATTN_CORE', 'true').lower() == 'true',
            'attn_upproj': os.getenv('RECOMPUTE_ATTN_UPPROJ', 'true').lower() == 'true',
        }

        if self.component_switches['attn']:
            self.component_switches['attn_core'] = False
            self.component_switches['attn_upproj'] = False

        if self.component_switches['moe']:
            self.component_switches['router'] = False
            self.component_switches['permutation'] = False
            self.component_switches['experts'] = False
            self.component_switches['unpermutation'] = False
            self.component_switches['shared_experts'] = False
            self.component_switches['expert_fc1'] = False
            self.component_switches['expert_bias_act'] = False
            self.component_switches['expert_fc2'] = False

        if self.component_switches['experts']:
            self.component_switches['expert_fc1'] = False
            self.component_switches['expert_bias_act'] = False
            self.component_switches['expert_fc2'] = False
    
    def should_recompute(self, component_name: str) -> bool:
        """判断指定组件是否应该启用重计算"""
        return (
            self.global_enabled and 
            self.component_switches.get(component_name, False)
        )

# 全局重计算控制器实例
recompute_controller = RecomputeController()

def custom_recompute(component_name: str):
    """
    重计算装饰器，根据环境变量和组件名称决定是否启用重计算
    
    Args:
        component_name: 组件名称，用于查找对应的环境变量开关
    
    Usage:
        @custom_recompute('experts')
        def experts_forward(self, *args):
            return self.experts(*args)
            
        # 调用时按环境变量决定是否重计算
        result = experts_forward(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args):
            return conditional_checkpoint(component_name, False, func, *args)
        
        return wrapper
    return decorator

def conditional_checkpoint(component_name: str, force_recompute: bool, func: Callable, *args):
    """
    条件重计算helper函数，便于在调用处控制重计算
    
    Args:
        component_name: 组件名称
        force_recompute: 是否强制重计算
        func: 要调用的函数
        *args: 传递给函数的参数
        
    Usage:
        # 在forward方法中使用
        output = conditional_checkpoint('expert_fc1', self.linear_fc1, input_tensor, tokens_per_expert)
    """
    # 根据环境变量决定是否启用重计算
    should_recompute = recompute_controller.should_recompute(component_name) or force_recompute
    
    if not should_recompute:
        # 不启用重计算，直接调用原函数
        return func(*args)
    
    # 启用重计算，统一使用te_checkpoint
    from megatron.core import tensor_parallel, parallel_state
    from megatron.core.extensions.transformer_engine import te_checkpoint

    return te_checkpoint(
        func,
        *args,
        distribute_saved_activations=False,
        get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    ) 