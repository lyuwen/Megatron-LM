import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import AutoConfig, AutoModel

from tqdm import tqdm
import transformers
from accelerate import init_empty_weights

import os
init_dir = '/mnt/cpfs/users/mzy/moe/output/hf_test'
init_path = '/mnt/cpfs/users/mzy/moe/output/hf_test/dsv2_normal_memory_init'
import sys
sys.path.insert(0, init_dir)
from init_hf_layer8.modeling_deepseek import DeepseekV3ForCausalLM
from init_hf_layer_pp_test_dsv2.modeling_deepseek import DeepseekV2ForCausalLM

config = AutoConfig.from_pretrained(init_path, trust_remote_code=True)
print(config)

model = DeepseekV2ForCausalLM(config)
print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters())
# pytorch_total_params / 1e9 # 
num_hidden_layers = model.config.num_hidden_layers
first_k_dense_replace = model.config.first_k_dense_replace
n_routed_experts = model.config.n_routed_experts
num_experts_per_tok = model.config.num_experts_per_tok
hidden_size = model.config.hidden_size
moe_intermediate_size = model.config.moe_intermediate_size
# act = pytorch_total_params - (27-1) * (64-6) * 3 * 2048 * 1408
# act/1e9 # 2.66B
act = pytorch_total_params - (num_hidden_layers - first_k_dense_replace) * (n_routed_experts 
         - num_experts_per_tok) * 3 * hidden_size * moe_intermediate_size
print(pytorch_total_params / 1e9, act/1e9)

model.save_pretrained(init_path, safe_serialization=False)
