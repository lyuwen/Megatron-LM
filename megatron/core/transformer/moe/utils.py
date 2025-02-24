# Copyright (c) 2025, Lyuwen Fu. All rights reserved.

def get_mla_nparams(config) -> int:
    """ Calculate the number of parmeters for MLA layer
    """
    size_linear_proj = config.v_head_dim * config.num_attention_heads * config.hidden_size
    q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
    if config.q_lora_rank is not None:
        size_linear_q_proj = config.hidden_size * config.q_lora_rank + config.q_lora_rank * config.num_attention_heads * q_head_dim
    else:
        size_linear_q_proj = config.hidden_size * config.num_attention_heads * q_head_dim
    size_linear_kv_down_proj = config.hidden_size * (config.kv_lora_rank + config.qk_pos_emb_head_dim)
    size_linear_kv_up_proj = config.kv_lora_rank * config.num_attention_heads * (config.qk_head_dim + config.v_head_dim)
    return size_linear_proj + size_linear_q_proj + size_linear_kv_down_proj + size_linear_kv_up_proj


def get_dense_mlp_nparams(config) -> int:
    """ Calculate the number of parmeters for dense MLP layer
    """
    return (config.ffn_hidden_size * config.hidden_size) * 3


def get_shared_experts_nparams(config) -> int:
    """ Calculate the number of parmeters for the shared experts in the MoE MLP layer
    """
    return (config.moe_shared_expert_intermediate_size * config.hidden_size) * 3


def get_routed_experts_nparams(config) -> int:
    """ Calculate the number of parmeters for the routed experts in the MoE MLP layer
    """
    return (config.moe_ffn_hidden_size * config.hidden_size) * 3 * config.num_experts


def get_activated_experts_nparams(config) -> int:
    """ Calculate the number of parmeters for the activated experts in the MoE MLP layer
    """
    return (config.moe_ffn_hidden_size * config.hidden_size) * 3 * config.moe_router_topk


def get_dense_layer_size(config) -> int:
    """ Calculate the number of parmeters for the dense transformer layer
    """
    mla_size = get_mla_nparams(config)
    mlp_size = get_dense_mlp_nparams(config)
    return mla_size + mlp_size


def get_moe_total_layer_size(config) -> int:
    """ Calculate the number of parmeters for the MoE transformer layer
    """
    mla_size = get_mla_nparams(config)
    mlp_size = get_routed_experts_nparams(config) + get_shared_experts_nparams(config)
    return mla_size + mlp_size


def get_moe_activated_layer_size(config) -> int:
    """ Calculate the number of activated parmeters for the MoE transformer layer per batch
    """
    mla_size = get_mla_nparams(config)
    mlp_size = get_activated_experts_nparams(config) + get_shared_experts_nparams(config)
    return mla_size + mlp_size


def get_moe_layer_FLOPs(config, batch_size: int) -> int:
    """ Calculate the number of floating point operators for the MoE transformer layer
    """
    return 6 * batch_size * config.seq_length * get_moe_activated_layer_size(config)


def get_moe_model_size(config) -> int:
    dense_size = get_dense_layer_size(config)
    moe_size = get_moe_total_layer_size(config)
    if isinstance(config.moe_layer_freq, int):
        return config.moe_layer_freq * dense_size + (config.num_layers - config.moe_layer_freq) * moe_size
    else:
        return sum([moe_size if flag else dense_size for flag in config.moe_layer_freq])


def get_moe_activated_size(config) -> int:
    dense_size = get_dense_layer_size(config)
    moe_size = get_moe_activated_layer_size(config)
    if isinstance(config.moe_layer_freq, int):
        return config.moe_layer_freq * dense_size + (config.num_layers - config.moe_layer_freq) * moe_size
    else:
        return sum([moe_size if flag else dense_size for flag in config.moe_layer_freq])


def get_embedding_size(config) -> int:
    return config.padded_vocab_size * config.hidden_size * 2


def get_moe_FLOPs(config, batch_size: int) -> int:
    """ Calculate the number of floating point operators for the whole MoE transformer decoder block per batch
    """
    return 6 * batch_size * config.seq_length * (get_moe_activated_size(config) + get_embedding_size(config))

