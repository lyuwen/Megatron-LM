import sys
import os
import re
import json
import torch
import math
from functools import partial
import copy
import safetensors
from collections import defaultdict, OrderedDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
# from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint # , load_sharded_checkpoint

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
from megatron.training.utils import get_ltor_masks_and_position_ids
from safetensors.torch import save_file

# path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# sys.path.append(os.path.join(path_dir, "examples"))
# from deepseek_v2.pretrain_deepseek import model_provider
# from megatron_patch.arguments import get_patch_args
# megatron_lm_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# sys.path.insert(0, megatron_lm_path)
from pretrain_gpt import model_provider

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

import numpy as np
from collections.abc import Mapping, Sequence

import pdb
from megatron.core import InferenceParams

@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device.
    """
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem

def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--target-decoder-last-pipeline-num-layers",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-num-layers-per-virtual-pipeline-stage",
        type=int,
        default=0
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    parser.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action='store_true',
    )

    parser.add_argument(
        "--convert-checkpoint-from-transformers-to-megatron",
        action='store_true',
    )

    parser.add_argument(
        "--iteration",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--use-low-memory-convert",
        action="store_true",
    )

    parser.add_argument(
        "--save-num-files",
        type=int,
        default=1 
    )

    parser.add_argument(
        "--check-diff",
        action="store_true",
    )

    return parser




def load_megatron_model_with_vp_support_unbalance(megatron_args):
    MG_PATH = megatron_args.load
    all_ckpts = sorted(os.listdir(MG_PATH))
    mg_state_dict_path = OrderedDict()
    for ckpts in all_ckpts:
        ckpt_path = os.path.join(MG_PATH, ckpts, "model_optim_rng.pt")
        mg_state_dict_path[ckpts] = ckpt_path

    print("> Loading Megatron checkpoint...")
    # DIR_NAME = "mp_rank_{tp:02}_{pp:03}_{ep:03}" if args.expert_model_parallel_size > 1 else "mp_rank_{tp:02}_{pp:03}}"
    args = torch.load(mg_state_dict_path[list(mg_state_dict_path.keys())[0]], map_location="cpu")["args"]
    DIR_NAME = "mp_rank_{tp:02}_{pp:03}_{ep:03}" if args.expert_model_parallel_size > 1 else "mp_rank_{tp:02}_{pp:03}"
    print(args)
    first_pp_layers = args.decoder_first_pipeline_num_layers
    last_pp_layers = args.decoder_last_pipeline_num_layers
    remained_stages = args.pipeline_model_parallel_size
    remained_layers = args.num_layers
    if first_pp_layers:
        remained_layers -= first_pp_layers
        remained_stages -= 1
    if last_pp_layers:
        remained_layers -= last_pp_layers
        remained_stages -= 1
    mid_pp_layers = remained_layers // remained_stages
    pp_layers_per_stage = ([first_pp_layers] if first_pp_layers else []) + [mid_pp_layers] * remained_stages + \
                          ([last_pp_layers] if last_pp_layers else [])
    if args.num_virtual_stages_per_pipeline_rank:
        virtual_pipeline_stage_size = args.num_virtual_stages_per_pipeline_rank
    else:
        virtual_pipeline_stage_size = args.num_layers // (
                    args.num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size)
    print(pp_layers_per_stage)
    print(virtual_pipeline_stage_size)
    num_local_experts = args.num_experts // args.expert_model_parallel_size
    mid_state = defaultdict(list)
    for tp_rank in range(args.tensor_model_parallel_size):
        for ep_rank in range(args.expert_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                for vp_rank in range(virtual_pipeline_stage_size):
                    curr_ckpt = DIR_NAME.format(tp=tp_rank, pp=pp_rank,
                                                ep=ep_rank) if args.expert_model_parallel_size > 1 else DIR_NAME.format(
                        tp=tp_rank, pp=pp_rank)
                    vp_model = "model{}".format(vp_rank)
                    # model_state_dict = mg_state_dict[curr_ckpt][vp_model]
                    model_state_dict = torch.load(mg_state_dict_path[curr_ckpt], map_location="cpu")[vp_model]

                    # combine mg model first
                    for k, v in model_state_dict.items():
                        if "_extra_state" in k:
                            continue

                        if "decoder.layers." in k:
                            # map to global layer_id
                            local_layer_id = int(re.search(r"decoder\.layers\.(\d+)\.", k).group(1))
                            # layer_id = (pp_rank * virtual_pipeline_stage_size + vp_rank) * args.num_layers_per_virtual_pipeline_stage + local_layer_id
                            #layer_id = args.num_layers_per_virtual_pipeline_stage * (
                            #            vp_rank * args.pipeline_model_parallel_size + pp_rank) + local_layer_id  # ref https://github.com/NVIDIA/Megatron-LM/blob/e40f0f9abc96e5237906b5c668baf36bed7563fd/megatron/core/parallel_state.py#L467
                            layer_id = sum(pp_layers_per_stage[:pp_rank])+ vp_rank * pp_layers_per_stage[pp_rank] // virtual_pipeline_stage_size  + local_layer_id
                            new_k = re.sub(r"decoder.layers.\d+", "decoder.layers.{}".format(layer_id), k)

                            # map to global expert_id
                            if "local_experts" in k:
                                local_expert_rank = int(re.search(r"local_experts\.(\d+)\.", k).group(1))
                                expert_rank = ep_rank * num_local_experts + local_expert_rank
                                new_k = re.sub(r"local_experts\.\d+", "local_experts.{}".format(expert_rank), new_k)
                                mid_state[new_k].append(v)
                            elif "mlp.experts" in k:  # group gemm
                                local_expert_rank = int(
                                    re.search(r"mlp\.experts\.linear_fc[12]\.weight(\d+)", k).group(1))
                                expert_rank = ep_rank * num_local_experts + local_expert_rank
                                new_k = re.sub(r"weight\d+", "weight{}".format(expert_rank), new_k)
                                mid_state[new_k].append(v)
                            else:
                                if ep_rank == 0:
                                    mid_state[new_k].append(v)
                        else:
                            if "word_embeddings" in k:
                                if ep_rank == 0 and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank == 0 and pp_rank == args.pipeline_model_parallel_size - 1 and vp_rank == virtual_pipeline_stage_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing")

    print("> Combining dist-checkpoint...")
    del model_state_dict
    combined_state_dict = {}
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
    for k, v in mid_state.items():
        if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
            target_v = v[0]
        elif 'word_embeddings' in k or 'output_layer' in k or 'final_layernorm' in k:
            target_v = torch.cat(v, dim=0)
        elif 'linear_proj' in k:
            target_v = torch.cat(v, dim=1)
        elif 'linear_q_proj' in k:
            viewed = [x.view(group_per_split, -1, q_head_dim, args.hidden_size) for x in v]
            target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
        elif 'linear_kv_b_proj' in k or 'linear_kv_up_proj.weight' in k:
            viewed = [
                x.view(group_per_split, -1, q_head_dim - args.qk_pos_emb_head_dim + args.v_head_dim, args.kv_lora_rank)
                for x in v]
            target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
        elif 'linear_kv_up_proj.layer_norm_weight' in k:
            target_v = v[0]
        elif 'linear_q_b_proj' in k or 'linear_q_up_proj' in k:  #
            target_v = v[0]
        elif 'linear_q_a_proj' in k or 'linear_q_down_proj' in k:  #
            target_v = v[0]
        elif 'linear_kv_a_proj' in k or 'linear_kv_down_proj' in k:
            target_v = v[0]
        elif 'linear_fc1.weight' in k:
            viewed = [x.view(2, -1, args.hidden_size) for x in v]
            target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
        elif 'linear_fc1.layer_norm_weight' in k:
            target_v = v[0]
        elif 'linear_fc2' in k:
            target_v = torch.cat(v, dim=1)
        elif 'input_layernorm' in k:
            target_v = v[0]
        elif 'q_a_layernorm' in k or 'q_layernorm' in k:  #
            target_v = v[0]
        elif 'kv_a_layernorm' in k or 'kv_layernorm' in k:
            target_v = v[0]
        elif 'pre_mlp_layernorm' in k:
            target_v = v[0]
        else:
            raise ValueError(f"{k} is missing!")
        combined_state_dict[k] = target_v

    print("> Loading checkpoint into model...")
    del mid_state
    model = model_provider()
    model.load_state_dict(combined_state_dict, strict=False)
    return model

# def name_to_expert_rank(key):
#     # pattern = r'local_experts\.(\d+)\.'
#     pattern = r'mlp.experts.linear_fc[12].weight(\d+)'
#     expert_rank = int(re.findall(pattern, key)[0])
#     return expert_rank


def load_megatron_model_with_vp(megatron_args):
    MG_PATH = megatron_args.load
    all_ckpts = sorted(os.listdir(MG_PATH))
    mg_state_dict_path = OrderedDict()
    for ckpts in all_ckpts:
        ckpt_path = os.path.join(MG_PATH, ckpts, "model_optim_rng.pt")
        mg_state_dict_path[ckpts] = ckpt_path
    
    print("> Loading Megatron checkpoint...")
    #DIR_NAME = "mp_rank_{tp:02}_{pp:03}_{ep:03}" if args.expert_model_parallel_size > 1 else "mp_rank_{tp:02}_{pp:03}}"
    args = torch.load(mg_state_dict_path[list(mg_state_dict_path.keys())[0]], map_location="cpu")["args"]
    DIR_NAME = "mp_rank_{tp:02}_{pp:03}_{ep:03}" if args.expert_model_parallel_size > 1 else "mp_rank_{tp:02}_{pp:03}"
    print(args)
    virtual_pipeline_stage_size = args.num_layers // (args.num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size)
    num_local_experts = args.num_experts // args.expert_model_parallel_size
    mid_state = defaultdict(list)
    for tp_rank in range(args.tensor_model_parallel_size):
        for ep_rank in range(args.expert_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                for vp_rank in range(virtual_pipeline_stage_size):
                    print(DIR_NAME)
                    curr_ckpt = DIR_NAME.format(tp=tp_rank, pp=pp_rank, ep=ep_rank) if args.expert_model_parallel_size > 1 else DIR_NAME.format(tp=tp_rank, pp=pp_rank)
                    vp_model = "model{}".format(vp_rank)
                    # model_state_dict = mg_state_dict[curr_ckpt][vp_model]
                    model_state_dict = torch.load(mg_state_dict_path[curr_ckpt], map_location="cpu")[vp_model]

                    # combine mg model first
                    for k, v in model_state_dict.items():
                        if "_extra_state" in k:
                            continue

                        if "decoder.layers." in k:
                            # map to global layer_id
                            local_layer_id = int(re.search(r"decoder\.layers\.(\d+)\.", k).group(1))
                            # layer_id = (pp_rank * virtual_pipeline_stage_size + vp_rank) * args.num_layers_per_virtual_pipeline_stage + local_layer_id
                            layer_id = args.num_layers_per_virtual_pipeline_stage * (vp_rank * args.pipeline_model_parallel_size + pp_rank) + local_layer_id # ref https://github.com/NVIDIA/Megatron-LM/blob/e40f0f9abc96e5237906b5c668baf36bed7563fd/megatron/core/parallel_state.py#L467
                            new_k = re.sub(r"decoder.layers.\d+", "decoder.layers.{}".format(layer_id), k)

                            # map to global expert_id
                            if "local_experts" in k:
                                local_expert_rank = int(re.search(r"local_experts\.(\d+)\.", k).group(1))
                                expert_rank = ep_rank * num_local_experts + local_expert_rank
                                new_k = re.sub(r"local_experts\.\d+", "local_experts.{}".format(expert_rank), new_k)
                                mid_state[new_k].append(v)
                            elif "mlp.experts" in k: # group gemm
                                local_expert_rank = int(re.search(r"mlp\.experts\.linear_fc[12]\.weight(\d+)", k).group(1))
                                expert_rank = ep_rank * num_local_experts + local_expert_rank
                                new_k = re.sub(r"weight\d+", "weight{}".format(expert_rank), new_k)
                                mid_state[new_k].append(v)
                            else:
                                if ep_rank == 0:
                                    mid_state[new_k].append(v)
                        else:
                            if "word_embeddings" in k:
                                if ep_rank == 0 and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank == 0 and pp_rank == args.pipeline_model_parallel_size - 1 and vp_rank == virtual_pipeline_stage_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing")
    
    print("> Combining dist-checkpoint...")
    del model_state_dict
    combined_state_dict = {}
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
    for k, v in mid_state.items():
        if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
            target_v = v[0]
        elif 'word_embeddings' in k or 'output_layer' in k or 'final_layernorm' in k:
            target_v = torch.cat(v, dim=0)
        elif 'linear_proj' in k:
            target_v = torch.cat(v, dim=1)
        elif 'linear_q_proj' in k:
            viewed = [x.view(group_per_split, -1, q_head_dim, args.hidden_size) for x in v]
            target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
        elif 'linear_kv_b_proj' in k or 'linear_kv_up_proj.weight' in k:
            viewed = [x.view(group_per_split, -1, q_head_dim - args.qk_pos_emb_head_dim + args.v_head_dim, args.kv_lora_rank) for x in v]
            target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
        elif 'linear_kv_up_proj.layer_norm_weight' in k:
            target_v = v[0]
        elif 'linear_q_b_proj' in k or 'linear_q_up_proj' in k: #
            target_v = v[0]
        elif 'linear_q_a_proj' in k or 'linear_q_down_proj' in k: #
            target_v = v[0]
        elif 'linear_kv_a_proj' in k or 'linear_kv_down_proj' in k:
            target_v = v[0]
        elif 'linear_fc1.weight' in k:
            viewed = [x.view(2, -1, args.hidden_size) for x in v]
            target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
        elif 'linear_fc1.layer_norm_weight' in k:
            target_v = v[0]
        elif 'linear_fc2' in k:
            target_v = torch.cat(v, dim=1)
        elif 'input_layernorm' in k:
            target_v = v[0]
        elif 'q_a_layernorm' in k or 'q_layernorm' in k: #
            target_v = v[0]
        elif 'kv_a_layernorm' in k or 'kv_layernorm' in k:
            target_v = v[0]
        elif 'pre_mlp_layernorm' in k:
            target_v = v[0]
        else:
            raise ValueError(f"{k} is missing!")
        combined_state_dict[k] = target_v

    print("> Loading checkpoint into model...")
    del mid_state
    model = model_provider()
    model.load_state_dict(combined_state_dict, strict=False)
    return model


def load_megatron_model_latest(args):
    #os.makedirs(args.save, exist_ok=True)
    #os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.save)
    #os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.save)
    #os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.save)
    #os.system("cp -rf " + args.hf_ckpt_path + "/special_tokens_map.json " + args.save)

    # os.system("cp -rf " + args.hf_ckpt_path + "/*config.json " + args.load)
    # os.system("cp -rf " + args.hf_ckpt_path + "/tokenizer* " + args.load)
    # os.system("cp -rf " + args.hf_ckpt_path + "/*.py " + args.load)
    # os.system("cp -rf " + args.hf_ckpt_path + "/special_tokens_map.json " + args.load)

    # os.system(f"cp ./modeling_deepseek_align_version.py {args.save}/modeling_deepseek.py")  # replace the modeling file - no rescale for pretrain temp.

    model_path = args.load
    if args.iteration < 0:
        tracker_filename = get_checkpoint_tracker_filename(model_path)
        iteration, release = read_metadata(tracker_filename)
    else:
        iteration = args.iteration
        release = False

    iter_folder = os.path.join(model_path, f'iter_{iteration:07d}') if not release else f'{model_path}/release'

    if args.target_num_layers_per_virtual_pipeline_stage: # or args.num_virtual_stages_per_pipeline_rank:
        #args.num_layers_per_virtual_pipeline_stage = args.target_num_layers_per_virtual_pipeline_stage
        args.load = iter_folder
        return load_megatron_model_with_vp_support_unbalance(args)

    model = model_provider()

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    
    q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
    group_per_split = args.num_attention_heads // args.tensor_model_parallel_size
    if args.num_experts is not None:
        pattern_exp = r'mlp\.experts\.linear_fc[12]\.weight(\d+)' if args.moe_grouped_gemm else r'local_experts\.(\d+)\.'
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)
    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, None, None)
        state_dict = torch.load(checkpoint_name)['model']
    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.expert_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, True, ep_rank)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
            for k, v in split_state.items():
                if 'local_experts' in k and "_extra_state" not in k:
                    expert_local_rank = int(re.findall(pattern_exp, k)[0])
                    expert_rank = expert_local_rank + num_local_experts * ep_rank
                    k = k.replace(f'local_experts.{expert_local_rank}', f'local_experts.{expert_rank}')
                elif 'mlp.experts' in k and "_extra_state" not in k:
                    expert_local_rank = int(re.findall(pattern_exp, k)[0])
                    expert_rank = expert_local_rank + num_local_experts * ep_rank
                    k = k.replace(f'weight{expert_local_rank}', f'weight{expert_rank}')
                if k not in state_dict.keys():
                    state_dict[k] = v
    elif (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size >= 1
        and args.expert_model_parallel_size >= 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):  
        first_pp_layers = args.target_decoder_first_pipeline_num_layers
        last_pp_layers = args.target_decoder_last_pipeline_num_layers
        remained_stages = args.pipeline_model_parallel_size
        remained_layers = args.num_layers
        if first_pp_layers:
            remained_layers -= first_pp_layers
            remained_stages -= 1
        if last_pp_layers:
            remained_layers -= last_pp_layers
            remained_stages -= 1
        assert remained_layers % remained_stages == 0
        mid_pp_layers = remained_layers // remained_stages
        pp_layers_per_stage = ([first_pp_layers] if first_pp_layers else []) + [mid_pp_layers] * remained_stages + \
                                                                       ( [last_pp_layers] if last_pp_layers else [])
        #assert args.num_layers % args.pipeline_model_parallel_size == 0 # temp
        # if args.target_decoder_first_pipeline_num_layers is not None:
        #     remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
        #     remained_stages = args.pipeline_model_parallel_size - 1
        #     assert remained_layers % remained_stages == 0
        #     pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)
        # else:
        #pp_layers_per_stage = [args.num_layers // args.pipeline_model_parallel_size] * args.pipeline_model_parallel_size
        # when not divisible
        #num_layers = args.num_layers // args.pipeline_model_parallel_size
        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = sum(pp_layers_per_stage[:pp_rank])
                    for layer in range(pp_layers_per_stage[pp_rank]):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[(pp_rank, layer)] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                    for k, v in split_state.items():
                        if '_extra_state' in k:
                            continue
                        if 'decoder.layers' in k:
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy[(pp_rank, int(res[0]))]), k)
                            if 'local_experts' in k:
                                local_expert_rank = int(re.findall(pattern_exp, tgt)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank ##
                                tgt = tgt.replace(f'local_experts.{local_expert_rank}', f'local_experts.{expert_rank}')
                                mid_state[tgt].append(v)
                            elif 'mlp.experts' in k:
                                local_expert_rank = int(re.findall(pattern_exp, tgt)[0])
                                expert_rank = local_expert_rank + num_local_experts * ep_rank ##
                                tgt = tgt.replace(f'weight{local_expert_rank}', f'weight{expert_rank}')
                                mid_state[tgt].append(v)
                            else:
                                # if 'linear_proj' in k or 'linear_q_proj' in k or 'linear_q_down_proj' in k or 'linear_q_up_proj'in k or \
                                #         'linear_kv_up_proj' in k or 'linear_kv_down_proj' in k or 'mlp.linear_fc2' in k or \
                                #         'mlp.linear_fc1' in k or 'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k or \
                                #         'linear_kv_a_proj' in k or 'linear_kv_b_proj' in k or 'linear_q_a_proj' in k or 'linear_q_b_proj' in k or \
                                #         'shared_expert.linear_fc1' in k or 'shared_expert.linear_fc2' in k:
                                if ep_rank ==0:
                                    mid_state[tgt].append(v)
                                # else:
                                #     mid_state[tgt].append(v)
                        else:
                            if "word_embeddings" in k:
                                if ep_rank ==0 and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank ==0 and pp_rank == args.pipeline_model_parallel_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError(f"{k} is missing! ")

        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif 'word_embeddings' in k or 'output_layer' in k or 'final_layernorm' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_q_proj' in k:
                viewed = [x.view(group_per_split, -1, q_head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_kv_up_proj.weight' in k or 'linear_kv_b_proj' in k:
                viewed = [x.view(group_per_split, -1, q_head_dim - args.qk_pos_emb_head_dim + args.v_head_dim, args.kv_lora_rank) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.kv_lora_rank)
            elif 'linear_kv_up_proj.layer_norm_weight' in k:
                target_v = v[0]
            elif 'linear_q_up_proj' in k or 'linear_q_b_proj' in k:
                target_v = v[0]
            elif 'linear_q_down_proj' in k or 'linear_q_a_proj' in k:
                target_v = v[0]
            elif 'linear_kv_down_proj' in k or 'linear_kv_a_proj' in k:
                target_v = v[0]
            elif 'linear_fc1.weight' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif 'linear_fc1.layer_norm_weight' in k:
                target_v = v[0]
            elif 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'input_layernorm' in k:
                target_v = v[0]
            elif 'q_layernorm' in k or 'q_a_layernorm' in k:
                target_v = v[0]
            elif 'kv_layernorm' in k or 'kv_a_layernorm' in k:
                target_v = v[0]
            elif 'pre_mlp_layernorm' in k:
                target_v = v[0]
            else:
                raise ValueError(f"{k} is missing!")
            state_dict[k] = target_v

    else:
        raise ValueError('not support yet')
    #for k,v in model.named_parameters():
    #  if k not in state_dict:
    #    print(k)
    #print("check potential missed keys")

    model.load_state_dict(state_dict, strict=False)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):
    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()
    first_k_dense_replace = args.moe_first_k_dense_replace

    with torch.no_grad():
        hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
            hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)
            # hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
            #if layer_idx <= first_k_dense_replace - 1 and args.moe_grouped_gemm:
            if layer_idx <= first_k_dense_replace - 1 :
                hflayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight)
            else:
                hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

            if args.q_lora_rank is not None:
                hflayer.self_attn.q_a_proj.weight.copy_(mglayer.self_attention.linear_q_down_proj.weight)
                hflayer.self_attn.q_b_proj.weight.copy_(mglayer.self_attention.linear_q_up_proj.weight)
                hflayer.self_attn.q_a_layernorm.weight.copy_(mglayer.self_attention.linear_q_up_proj.layer_norm_weight)
            else:
                hflayer.self_attn.q_proj.weight.copy_(mglayer.self_attention.linear_q_proj.weight)
            # hflayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_a_proj_with_mqa.weight)
            hflayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_down_proj.weight)
            # hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_b_proj.weight)
            hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_up_proj.weight)
            # hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.kv_a_layernorm.weight)
            hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.linear_kv_up_proj.layer_norm_weight)
            hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

            if layer_idx <= first_k_dense_replace - 1:
                gate_weight, up_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
                hflayer.mlp.gate_proj.weight.copy_(gate_weight)
                hflayer.mlp.up_proj.weight.copy_(up_weight)
                hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)

            else:
                hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
                # expert bias in Deepseek V3
                if hasattr(mglayer.mlp.router, 'expert_bias') and mglayer.mlp.router.expert_bias is not None:
                    print("e_score_correction_bias weight copy")
                    hflayer.mlp.gate.e_score_correction_bias.copy_(mglayer.mlp.router.expert_bias)

                if not args.moe_grouped_gemm:
                    for mgexpert, hfexpert in zip(mglayer.mlp.experts.local_experts, hflayer.mlp.experts):
                        gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight,
                                                            split_size_or_sections=args.moe_ffn_hidden_size)
                        hfexpert.gate_proj.weight.copy_(gate_weight)
                        hfexpert.up_proj.weight.copy_(up_weight)
                        hfexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)
                else:
                    for expert_id, hfexpert in enumerate(hflayer.mlp.experts):
                        mgexpert_linear_fc1_weight = getattr(mglayer.mlp.experts.linear_fc1, f"weight{expert_id}")
                        mgexpert_linear_fc2_weight = getattr(mglayer.mlp.experts.linear_fc2, f"weight{expert_id}")
                        gate_weight, up_weight = torch.split(mgexpert_linear_fc1_weight,
                                                            split_size_or_sections=args.moe_ffn_hidden_size)
                        hfexpert.gate_proj.weight.copy_(gate_weight)
                        hfexpert.up_proj.weight.copy_(up_weight)
                        hfexpert.down_proj.weight.copy_(mgexpert_linear_fc2_weight)

                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_experts.linear_fc1.weight,
                                split_size_or_sections=args.moe_shared_expert_intermediate_size) # args.moe_ffn_hidden_size*args.num_shared_experts
                hflayer.mlp.shared_experts.gate_proj.weight.copy_(shared_expert_gate_weight)
                hflayer.mlp.shared_experts.up_proj.weight.copy_(shared_expert_up_weight)
                hflayer.mlp.shared_experts.down_proj.weight.copy_(mglayer.mlp.shared_experts.linear_fc2.weight)

        hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def set_state_dict(sd,key,val):
    sd[key] = val



def convert_checkpoint_from_megatron_to_transformers_low_memory(mgmodel, args):
    if args.fp16:
        mgmodel = mgmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
    SAVE_NAME = "model-{now_part:05}-of-{total_part:05}.safetensors"
    total_files = args.save_num_files
    now_files = 1
    layer_to_save = int(math.ceil(args.num_layers/total_files))
    first_k_dense_replace = args.moe_first_k_dense_replace
    state_dict = {}
    saved_key_index = {}
    with torch.no_grad():
        #hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        set_state_dict(state_dict,"model.embed_tokens.weight",mgmodel.embedding.word_embeddings.weight)
        for layer_idx, mglayer in enumerate(mgmodel.decoder.layers):
            layer_prefix = f"model.layers.{layer_idx}"
            #hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)
            set_state_dict(state_dict,f"{layer_prefix}.input_layernorm.weight",mglayer.input_layernorm.weight)
            # hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
            if layer_idx <= first_k_dense_replace - 1:
                #hflayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight)
                set_state_dict(state_dict,f"{layer_prefix}.post_attention_layernorm.weight",mglayer.mlp.linear_fc1.layer_norm_weight)
            else:
                #hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
                set_state_dict(state_dict,f"{layer_prefix}.post_attention_layernorm.weight",mglayer.pre_mlp_layernorm.weight)

            if args.q_lora_rank is not None:
                #hflayer.self_attn.q_a_proj.weight.copy_(mglayer.self_attention.linear_q_down_proj.weight)
                set_state_dict(state_dict, f"{layer_prefix}.self_attn.q_a_proj.weight",
                               mglayer.self_attention.linear_q_down_proj.weight)
                #hflayer.self_attn.q_b_proj.weight.copy_(mglayer.self_attention.linear_q_up_proj.weight)
                set_state_dict(state_dict, f"{layer_prefix}.self_attn.q_b_proj.weight",
                               mglayer.self_attention.linear_q_up_proj.weight)
                #hflayer.self_attn.q_a_layernorm.weight.copy_(mglayer.self_attention.linear_q_up_proj.layer_norm_weight)
                set_state_dict(state_dict, f"{layer_prefix}.self_attn.q_a_layernorm.weight",
                               mglayer.self_attention.linear_q_up_proj.layer_norm_weight)
            else:
                #hflayer.self_attn.q_proj.weight.copy_(mglayer.self_attention.linear_q_proj.weight)
                set_state_dict(state_dict, f"{layer_prefix}.self_attn.q_proj.weight",
                               mglayer.self_attention.linear_q_proj.weight)
            #hflayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_down_proj.weight)
            set_state_dict(state_dict, f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.weight",
                           mglayer.self_attention.linear_kv_down_proj.weight)
            # hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_b_proj.weight)
            #hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_up_proj.weight)
            set_state_dict(state_dict, f"{layer_prefix}.self_attn.kv_b_proj.weight",
                           mglayer.self_attention.linear_kv_up_proj.weight)
            # hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.kv_a_layernorm.weight)
            #hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.linear_kv_up_proj.layer_norm_weight)
            set_state_dict(state_dict, f"{layer_prefix}.self_attn.kv_a_layernorm.weight",
                           mglayer.self_attention.linear_kv_up_proj.layer_norm_weight)
            #hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)
            set_state_dict(state_dict, f"{layer_prefix}.self_attn.o_proj.weight",
                           mglayer.self_attention.linear_proj.weight)

            if layer_idx <= first_k_dense_replace - 1:
                gate_weight, up_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
                #hflayer.mlp.gate_proj.weight.copy_(gate_weight)
                set_state_dict(state_dict, f"{layer_prefix}.mlp.gate_proj.weight",
                               gate_weight)
                #hflayer.mlp.up_proj.weight.copy_(up_weight)
                set_state_dict(state_dict, f"{layer_prefix}.mlp.up_proj.weight",
                               up_weight)
                #hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)
                set_state_dict(state_dict, f"{layer_prefix}.mlp.down_proj.weight",
                               mglayer.mlp.linear_fc2.weight)

            else:
                #hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
                set_state_dict(state_dict, f"{layer_prefix}.mlp.gate.weight",
                               mglayer.mlp.router.weight)
                # expert bias in Deepseek V3
                if hasattr(mglayer.mlp.router, 'expert_bias') and mglayer.mlp.router.expert_bias is not None:
                    #hflayer.mlp.gate.e_score_correction_bias.copy_(mglayer.mlp.router.expert_bias)
                    set_state_dict(state_dict, f"{layer_prefix}.mlp.gate.e_score_correction_bias",
                                   mglayer.mlp.router.expert_bias)

                if not args.moe_grouped_gemm:
                    for expert_id,mgexpert in enumerate(mglayer.mlp.experts.local_experts):
                        layer_expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
                        gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight,
                                                             split_size_or_sections=args.moe_ffn_hidden_size)
                        #hfexpert.gate_proj.weight.copy_(gate_weight)
                        set_state_dict(state_dict, f"{layer_expert_prefix}.gate_proj.weight",
                                       gate_weight)
                        #hfexpert.up_proj.weight.copy_(up_weight)
                        set_state_dict(state_dict, f"{layer_expert_prefix}.up_proj.weight",
                                       up_weight)
                        #hfexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)
                        set_state_dict(state_dict, f"{layer_expert_prefix}.down_proj.weight",
                                       mgexpert.linear_fc2.weight)
                else:
                    for expert_id in range(args.num_experts):
                        layer_expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
                        mgexpert_linear_fc1_weight = getattr(mglayer.mlp.experts.linear_fc1, f"weight{expert_id}")
                        mgexpert_linear_fc2_weight = getattr(mglayer.mlp.experts.linear_fc2, f"weight{expert_id}")
                        gate_weight, up_weight = torch.split(mgexpert_linear_fc1_weight,
                                                            split_size_or_sections=args.moe_ffn_hidden_size)
                        #hfexpert.gate_proj.weight.copy_(gate_weight)
                        set_state_dict(state_dict, f"{layer_expert_prefix}.gate_proj.weight",
                                       gate_weight)
                        #hfexpert.up_proj.weight.copy_(up_weight)
                        set_state_dict(state_dict, f"{layer_expert_prefix}.up_proj.weight",
                                       up_weight)
                        #hfexpert.down_proj.weight.copy_(mgexpert_linear_fc2_weight)
                        set_state_dict(state_dict, f"{layer_expert_prefix}.down_proj.weight",
                                       mgexpert_linear_fc2_weight)

                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_experts.linear_fc1.weight,
                                split_size_or_sections=args.moe_shared_expert_intermediate_size) # args.moe_ffn_hidden_size*args.num_shared_experts
                #hflayer.mlp.shared_experts.gate_proj.weight.copy_(shared_expert_gate_weight)
                set_state_dict(state_dict, f"{layer_prefix}.mlp.shared_experts.gate_proj.weight",
                               shared_expert_gate_weight)
                #hflayer.mlp.shared_experts.up_proj.weight.copy_(shared_expert_up_weight)
                set_state_dict(state_dict, f"{layer_prefix}.mlp.shared_experts.up_proj.weight",
                               shared_expert_up_weight)
                #hflayer.mlp.shared_experts.down_proj.weight.copy_(mglayer.mlp.shared_experts.linear_fc2.weight)
                set_state_dict(state_dict, f"{layer_prefix}.mlp.shared_experts.down_proj.weight",
                               mglayer.mlp.shared_experts.linear_fc2.weight)
            if (layer_idx + 1) % layer_to_save == 0 and now_files != total_files:
                save_name = SAVE_NAME.format(now_part=now_files,total_part=total_files)
                for k in state_dict.keys():
                    saved_key_index[k] = save_name
                save_file(state_dict,f"{args.save}/{save_name}")
                state_dict = {}
                now_files +=1
        #hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        set_state_dict(state_dict, "model.norm.weight",
                       mgmodel.decoder.final_layernorm.weight)
        #hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)
        set_state_dict(state_dict,"lm_head.weight",
                       mgmodel.output_layer.weight)
        save_name = SAVE_NAME.format(now_part=now_files,total_part=total_files)
        for k in state_dict.keys():
            saved_key_index[k] = save_name
        save_file(state_dict, f"{args.save}/{save_name}")
        #if total_files > 1:
        with open(f"{args.save}/model.safetensors.index.json","w") as f:
            to_save = {"weight_map":saved_key_index,"metadata":{}}
            json.dump(to_save,f)





def convert_checkpoint_from_megatron_to_transformers_with_vp(mgmodel, hfmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    with torch.no_grad():
        hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
            hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)
            hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)
            if args.q_lora_rank is not None:
                hflayer.self_attn.q_a_proj.weight.copy_(mglayer.self_attention.linear_q_down_proj.weight)
                hflayer.self_attn.q_b_proj.weight.copy_(mglayer.self_attention.linear_q_up_proj.weight)
                hflayer.self_attn.q_a_layernorm.weight.copy_(mglayer.self_attention.linear_q_up_proj.layer_norm_weight)
            else:
                hflayer.self_attn.q_proj.weight.copy_(mglayer.self_attention.linear_q_proj.weight)
            hflayer.self_attn.kv_a_proj_with_mqa.weight.copy_(mglayer.self_attention.linear_kv_a_proj_with_mqa.weight)
            hflayer.self_attn.kv_b_proj.weight.copy_(mglayer.self_attention.linear_kv_b_proj.weight)
            hflayer.self_attn.kv_a_layernorm.weight.copy_(mglayer.self_attention.kv_a_layernorm.weight)
            hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

            if layer_idx == -1: # since we also use moe in the first layer, we do not need this.
                gate_weight, up_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
                hflayer.mlp.gate_proj.weight.copy_(gate_weight)
                hflayer.mlp.up_proj.weight.copy_(up_weight)
                hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)

            else:
                hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)

                for mgexpert, hfexpert in zip(mglayer.mlp.experts.local_experts, hflayer.mlp.experts):
                    gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight,
                                                         split_size_or_sections=args.moe_ffn_hidden_size)
                    hfexpert.gate_proj.weight.copy_(gate_weight)
                    hfexpert.up_proj.weight.copy_(up_weight)
                    hfexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)

                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_expert.linear_fc1.weight,
                                split_size_or_sections=args.moe_ffn_hidden_size*args.num_shared_experts)
                hflayer.mlp.shared_experts.gate_proj.weight.copy_(shared_expert_gate_weight)
                hflayer.mlp.shared_experts.up_proj.weight.copy_(shared_expert_up_weight)
                hflayer.mlp.shared_experts.down_proj.weight.copy_(mglayer.mlp.shared_expert.linear_fc2.weight)

        hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)


def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):
    # to do : change key names later.
   # if args.fp16:
   #     mgmodel = mgmodel.half()
   #     hfmodel = hfmodel.half()
   # elif args.bf16:
   #     mgmodel = mgmodel.bfloat16()
   #     hfmodel = hfmodel.bfloat16()
    mgmodel = mgmodel.float()
    hfmodel = hfmodel.float()

    first_k_dense_replace = args.first_k_dense_replace

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)
        for layer_idx, (mglayer, hflayer) in enumerate(zip(mgmodel.decoder.layers, hfmodel.model.layers)):
            mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)
            mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)
            if args.q_lora_rank is not None:
                mglayer.self_attention.linear_q_a_proj.weight.copy_(hflayer.self_attn.q_a_proj.weight)
                mglayer.self_attention.linear_q_b_proj.weight.copy_(hflayer.self_attn.q_b_proj.weight)
                mglayer.self_attention.q_a_layernorm.weight.copy_(hflayer.self_attn.q_a_layernorm.weight)
            else:
                mglayer.self_attention.linear_q_proj.weight.copy_(hflayer.self_attn.q_proj.weight)
            mglayer.self_attention.linear_kv_a_proj_with_mqa.weight.copy_(hflayer.self_attn.kv_a_proj_with_mqa.weight)
            mglayer.self_attention.linear_kv_b_proj.weight.copy_(hflayer.self_attn.kv_b_proj.weight)
            mglayer.self_attention.kv_a_layernorm.weight.copy_(hflayer.self_attn.kv_a_layernorm.weight)
            mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)

            if layer_idx <= first_k_dense_replace - 1:
                mglayer.mlp.linear_fc1.weight.copy_(
                    torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight]))
                mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)
            else:
                mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight)
                for hf_expert, expert in zip(hflayer.mlp.experts, mglayer.mlp.experts.local_experts):
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    expert.linear_fc1.weight.copy_(fc1_weight)
                    expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)

                shared_fc1_weight = torch.cat(
                    [hflayer.mlp.shared_experts.gate_proj.weight, hflayer.mlp.shared_experts.up_proj.weight])
                mglayer.mlp.shared_expert.linear_fc1.weight.copy_(shared_fc1_weight)
                mglayer.mlp.shared_expert.linear_fc2.weight.copy_(hflayer.mlp.shared_experts.down_proj.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        if args.untie_embeddings_and_output_weights:
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


def save_state_dict(args, model, checkpoint_name):
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = 0
    state_dict['model'] = model
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)

def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()

def save_mgmodel(mgmodel, args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/*config.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    full_model = mgmodel.state_dict_for_save_checkpoint()

    for k in list(full_model.keys()):
        if full_model[k] is None or "_extra_state" in k:
            full_model.pop(k)

    if args.num_experts is not None:
        pattern = r'local_experts\.(\d+)\.'
        num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0

    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, full_model, checkpoint_name)
    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size >1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):

        for ep_rank in range(args.expert_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, None, None, True, ep_rank)
            print(f'save ep_rank {ep_rank} model to {checkpoint_name}')
            for k, v in full_model.items():
                if 'local_experts' in k:
                    expert_rank = int(re.findall(pattern, k)[0])
                    if expert_rank // num_local_experts != ep_rank:
                        continue
                    expert_local_rank = expert_rank % num_local_experts
                    k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                model_split[k] = v
            save_state_dict(args, model_split, checkpoint_name)
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                model_split = {}
                if args.expert_model_parallel_size >1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, True, ep_rank)
                elif args.expert_model_parallel_size ==1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, False)
                for k, v in full_model.items():
                    if not isinstance(v, torch.Tensor):
                        target_v = v
                    elif 'linear_q_proj' in k or 'linear_q_a_proj' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'linear_q_b_proj' in k:
                        seg_0 = v.shape[0] // args.tensor_model_parallel_size
                        seg_1 = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[seg_0 * tp_rank: seg_0 * (tp_rank + 1), seg_1 * tp_rank: seg_1 * (tp_rank + 1)]
                    elif 'q_a_layernorm' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'linear_kv_b_proj' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank:seg* (tp_rank + 1)]
                    elif 'linear_proj' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'embedding' in k or 'output_layer' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'decoder.layers.0.mlp.linear_fc2' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                    elif 'decoder.layers.0.mlp.linear_fc1' in k:
                        viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                        seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                        target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                    elif 'local_experts' in k:
                        expert_rank = int(re.findall(pattern, k)[0])
                        if expert_rank // num_local_experts != ep_rank:
                            continue
                        expert_local_rank = expert_rank % num_local_experts
                        if 'linear_fc1' in k and 'norm' not in k:
                            viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                            seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                    elif 'shared_expert' in k and 'gate' not in k:
                        if 'linear_fc1' in k:
                            viewed = v.view(-1, args.moe_ffn_hidden_size * args.num_shared_experts, args.hidden_size)
                            seg = args.moe_ffn_hidden_size * args.num_shared_experts // args.tensor_model_parallel_size
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                    else:
                        target_v = v
                    model_split[k] = target_v
                save_state_dict(args, model_split, checkpoint_name)

    elif (
        args.pipeline_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        assert args.num_layers % args.pipeline_model_parallel_size == 0
        num_layers = args.num_layers // args.pipeline_model_parallel_size
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    model_split = {}
                    layer_offset = pp_rank * num_layers
                    layers_to_copy = {}
                    for layer in range(num_layers):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, True, ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, False)
                    print(f'tensor_parallel & pipeline_parallel & expert_parallel, save model to {checkpoint_name}')
                    for k, v in full_model.items():
                        if check_layer(layers_to_copy, k):
                            layer_pattern = re.compile(r'\d+')
                            res = layer_pattern.findall(k)
                            k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                        elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k):
                            continue
                        if not isinstance(v, torch.Tensor):
                            target_v = v
                        elif 'linear_q_proj' in k or 'linear_q_a_proj' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'linear_q_b_proj' in k:
                            seg_0 = v.shape[0] // args.tensor_model_parallel_size
                            seg_1 = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[seg_0 * tp_rank: seg_0 * (tp_rank + 1), seg_1 * tp_rank: seg_1 * (tp_rank + 1)]
                        elif 'q_a_layernorm' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'linear_kv_b_proj' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank:seg * (tp_rank + 1)]
                        elif 'linear_proj' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'embedding' in k or 'output_layer' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'decoder.layers.0.mlp.linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'decoder.layers.0.mlp.linear_fc1' in k:
                            viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                            seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                        elif 'local_experts' in k:
                            expert_rank = int(re.findall(pattern, k)[0])
                            if expert_rank // num_local_experts != ep_rank:
                                continue
                            expert_local_rank = expert_rank % num_local_experts
                            if 'linear_fc1' in k:
                                viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                                seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                            k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                        elif 'shared_expert' in k and 'gate' not in k:
                            if 'linear_fc1' in k:
                                viewed = v.view(-1, args.moe_ffn_hidden_size * args.num_shared_experts,
                                                args.hidden_size)
                                seg = args.moe_ffn_hidden_size * args.num_shared_experts // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        else:
                            target_v = v
                        if "word_embeddings" in k:
                            if pp_rank == 0:
                                model_split[k] = target_v
                        elif "output_layer" in k or "final_layernorm" in k:
                            if pp_rank == args.pipeline_model_parallel_size - 1:
                                model_split[k] = target_v
                        else:
                            model_split[k] = target_v
                    save_state_dict(args, model_split, checkpoint_name)

    else:
        raise ValueError('Something is wrong, please check your tp/pp/ep size')

    print(f'megatron model is save to {args.save}')


def save_hfmodel(args, model):
    # output_state_dict = model.state_dict()
    #max_shard_size = "10GB"
    # shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    os.makedirs(args.save, exist_ok=True)
    # for shard_file, shard in shards.items():
    #     if args.save_safetensors:
    #         shard_file = shard_file.replace("pytorch_", "")
    #         shard_file = shard_file.replace(".bin", ".safetensors")
    #         target_file = os.path.join(args.save, shard_file)
    #         print(f'huggingface model is save to {target_file}')
    #         new_shard = {}
    #         for k, v in shard.items():
    #             new_shard[k] = copy.deepcopy(v)
    #         safetensors.torch.save_file(clone_state_dict(new_shard), target_file, metadata={"format": "pt"})
    #     else:
    #         target_file = os.path.join(args.save, shard_file)
    #         print(f'huggingface model is save to {target_file}')
    #         torch.save(clone_state_dict(shard), target_file)

    # if index is None:
    #     print(f"Model weights saved in {os.path.join(args.save, WEIGHTS_NAME)}")
    # else:
    #     save_index_file = os.path.join(args.save, WEIGHTS_INDEX_NAME)
    #     # Save the index as well
    #     new_index = index.copy()
    #     weight_map = {}
    #     for k, v in index["weight_map"].items():
    #         print(k, v)
    #         v = v.replace("pytorch_", "")
    #         v = v.replace(".bin", ".safetensors")
    #         weight_map[k] = v
    #     new_index.update({"weight_map": weight_map})
    #     with open(save_index_file, "w", encoding="utf-8") as f:
    #         content = json.dumps(new_index, indent=2, sort_keys=True) + "\n"
    #         f.write(content)
    #     print(
    #         f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
    #         f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
    #         f"index located at {save_index_file}."
    #     )
    print(args.save)
    print("saved")
    model.save_pretrained(args.save)

def check_hf_mg_forward(hfmodel, mgmodel, mgargs):
    hf_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    hidden_size = mgargs.hidden_size
    q_lora_rank = mgargs.q_lora_rank
    q_head_dim = mgargs.qk_head_dim + mgargs.qk_pos_emb_head_dim
    num_heads = mgargs.num_attention_heads
    v_head_dim = mgargs.v_head_dim
    vocab_size = mgargs.padded_vocab_size
    kv_a_dim = mgargs.kv_lora_rank + mgargs.qk_pos_emb_head_dim
    kv_b_dim = num_heads * (q_head_dim - mgargs.qk_pos_emb_head_dim + v_head_dim)
    kv_lora_rank = mgargs.kv_lora_rank
    first_k_dense_replace = mgargs.moe_first_k_dense_replace

    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hf':
            hf_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == 'mg' and 'layer' in mode and not 'layernorm' in mode: # 
            mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
        elif frame == 'mg':
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
        if mode in ['hf-lmhead']:
            hf_hiddens[layer_idx][name] = output.transpose(0, 1).reshape(-1, vocab_size)
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
            hf_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
        elif mode in ['mg-lmhead']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, vocab_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
            mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
        elif mode in ['hf-q_proj_out', 'hf-o_proj_out', 'hf-kv_b_proj_out', 'hf-kv_a_proj_out', 'hf-kv_a_norm_out', 'hf-q_a_proj_out', 'hf-q_a_layernorm_out', 'hf-q_b_proj_out']:
            hf_hiddens[layer_idx][name] = output
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-kv_a_norm_out']:
            mg_hiddens[layer_idx][name] = output.reshape(-1, kv_lora_rank)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-q_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, num_heads * q_head_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-q_a_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, q_lora_rank)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        # elif mode in ['mg-q_a_layernorm_out']: # fused with q_up_proj, no output res.
        #     mg_hiddens[layer_idx][name] = output.reshape(-1, q_lora_rank)
        #     # mg_hiddens[layer_idx][name + '_weight'] = module.weight
        #     mg_hiddens[layer_idx][name + '_weight'] = module.layer_norm_weight
        elif mode in ['mg-q_b_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, num_heads * q_head_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-kv_a_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, kv_a_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-kv_b_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, kv_b_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-o_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, num_heads * v_head_dim)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hf-attn_out']:
            hf_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['mg-attn_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['hf-down_proj_out']:
            hf_hiddens[layer_idx][name] = output.reshape(-1, hidden_size)
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-down_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hf-shared_experts_down_proj_out']:
            hf_hiddens[layer_idx][name] = output.reshape(-1, hidden_size)
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-shared_experts_down_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hf-input_layernorm_out']:
            hf_hiddens[layer_idx][name] = output.flatten()
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
        elif mode in ["mg-input_layernorm_out"]:
            mg_hiddens[layer_idx][name] = output.flatten()
            mg_hiddens[layer_idx][name + "_weight"] = module.weight

    if mgargs.untie_embeddings_and_output_weights:
        hfmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hf-lmhead'),
                                            with_kwargs=True)
        mgmodel.output_layer.register_forward_hook(
            partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'), with_kwargs=True)

    for idx, layer in enumerate(hfmodel.model.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-layer_in'), with_kwargs=True)

        if mgargs.q_lora_rank is None:

            layer.self_attn.q_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-q_proj_in'),
                                                             with_kwargs=True)

            layer.self_attn.q_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-q_proj_out'),
                                                         with_kwargs=True)
            layer.input_layernorm.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-input_layernorm_in'),
                                                             with_kwargs=True)
            layer.input_layernorm.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-input_layernorm_out'),
                                                         with_kwargs=True)
        else:
            layer.self_attn.q_a_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-q_a_proj_in'), with_kwargs=True)
            layer.self_attn.q_a_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-q_a_proj_out'), with_kwargs=True)
            layer.self_attn.q_a_layernorm.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-q_a_layernorm_in'), with_kwargs=True)
            # layer.self_attn.q_a_layernorm.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-q_a_layernorm_out'), with_kwargs=True)
            # layer.self_attn.q_b_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-q_b_proj_in'), with_kwargs=True)
            layer.self_attn.q_b_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-q_b_proj_out'), with_kwargs=True)

        layer.self_attn.kv_a_proj_with_mqa.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='hf-kv_a_proj_in'), with_kwargs=True)

        layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='hf-kv_a_proj_out'), with_kwargs=True)

        # layer.self_attn.kv_a_layernorm.register_forward_pre_hook(
        #     partial(print_input_hook, layer_idx=idx, mode='hf-kv_a_norm_in'), with_kwargs=True)

        # layer.self_attn.kv_a_layernorm.register_forward_hook(
        #     partial(print_output_hook, layer_idx=idx, mode='hf-kv_a_norm_out'), with_kwargs=True)

        # layer.self_attn.kv_b_proj.register_forward_pre_hook(
        #     partial(print_input_hook, layer_idx=idx, mode='hf-kv_b_proj_in'), with_kwargs=True)

        # layer.self_attn.kv_b_proj.register_forward_hook(
        #     partial(print_output_hook, layer_idx=idx, mode='hf-kv_b_proj_out'), with_kwargs=True)

        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-o_proj_in'),
                                                         with_kwargs=True)

        layer.self_attn.o_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-o_proj_out'),
                                                     with_kwargs=True)

        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-attn_out'),
                                              with_kwargs=True)

        if idx <= first_k_dense_replace - 1:
            layer.mlp.down_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='hf-down_proj_in'), with_kwargs=True)

            layer.mlp.down_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='hf-down_proj_out'), with_kwargs=True)
        else:
            layer.mlp.shared_experts.down_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='hf-shared_experts_down_proj_in'), with_kwargs=True)

            layer.mlp.shared_experts.down_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='hf-shared_experts_down_proj_out'), with_kwargs=True)

    for idx, layer in enumerate(mgmodel.decoder.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)

        if mgargs.q_lora_rank is None:
            layer.self_attention.linear_q_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-q_proj_in'), with_kwargs=True)

            layer.self_attention.linear_q_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-q_proj_out'), with_kwargs=True)

            layer.input_layernorm.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-input_layernorm_in'), with_kwargs=True)

            layer.input_layernorm.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-input_layernorm_out'), with_kwargs=True)
        else:
            layer.self_attention.linear_q_down_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-q_a_proj_in'), with_kwargs=True)
            layer.self_attention.linear_q_down_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-q_a_proj_out'), with_kwargs=True)
            layer.self_attention.linear_q_up_proj.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-q_a_layernorm_in'), with_kwargs=True)
            # layer.self_attention.q_a_layernorm.register_forward_hook(
            #     partial(print_output_hook, layer_idx=idx, mode='mg-q_a_layernorm_out'), with_kwargs=True)
            # layer.self_attention.linear_q_b_proj.register_forward_pre_hook(
            #     partial(print_input_hook, layer_idx=idx, mode='mg-q_b_proj_in'), with_kwargs=True)
            layer.self_attention.linear_q_up_proj.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-q_b_proj_out'), with_kwargs=True)

        layer.self_attention.linear_kv_down_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-kv_a_proj_in'), with_kwargs=True)

        layer.self_attention.linear_kv_down_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-kv_a_proj_out'), with_kwargs=True)

        # layer.self_attention.kv_a_layernorm.register_forward_pre_hook(
        #     partial(print_input_hook, layer_idx=idx, mode='mg-kv_a_norm_in'), with_kwargs=True)

        # layer.self_attention.kv_a_layernorm.register_forward_hook(
        #     partial(print_output_hook, layer_idx=idx, mode='mg-kv_a_norm_out'), with_kwargs=True)

        # layer.self_attention.linear_kv_up_proj.register_forward_pre_hook(
        #     partial(print_input_hook, layer_idx=idx, mode='mg-kv_b_proj_in'), with_kwargs=True)

        # layer.self_attention.linear_kv_up_proj.register_forward_hook(
        #     partial(print_output_hook, layer_idx=idx, mode='mg-kv_b_proj_out'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)

        if idx <= first_k_dense_replace-1:
            layer.mlp.linear_fc2.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-down_proj_in'), with_kwargs=True)

            layer.mlp.linear_fc2.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-down_proj_out'), with_kwargs=True)
        else:
            layer.mlp.shared_experts.linear_fc2.register_forward_pre_hook(
                partial(print_input_hook, layer_idx=idx, mode='mg-shared_experts_down_proj_in'), with_kwargs=True)

            layer.mlp.shared_experts.linear_fc2.register_forward_hook(
                partial(print_output_hook, layer_idx=idx, mode='mg-shared_experts_down_proj_out'), with_kwargs=True)

    # input_ids = torch.tensor([[1, 2, 3]]).long().cuda()
    # input_ids = torch.randint(1,100,(1,100)).long().cuda()
    input_ids = torch.arange(100).unsqueeze(0).long().cuda()
    input_ids = torch.tensor([[128000,   2127,   6666,    734,    649,    387,   7633,    439,  13021,
            264,   3319,    323,    264,    743,    315,   1401,  19625,  13840,
            311,    459,   2612,     11,   1405,    279,   3319,     11,   7039,
             11,   2819,     11,    323,   2612,    527,    682,  23728,     13,
            578,   2612,    374]]).long().cuda() # zjllama tokenizer -> "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    inference_params = InferenceParams(max_batch_size=16, max_sequence_length=4096) ##
    print(hfmodel)
    print(mgmodel)
    is_oom = False
    with torch.inference_mode():
        # pdb.set_trace()
        try:
            hfmodel.cuda()
            hflogits = hfmodel(input_ids=input_ids).logits
        except torch.cuda.OutOfMemoryError:
            print('oom for huggingface model forward')
            is_oom = True
            
        hfmodel.cpu()
        del hfmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mglogits = mgmodel(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, inference_params=inference_params)
        except torch.cuda.OutOfMemoryError:
            print('oom for megatron model forward')
            is_oom = True
        mgmodel.cpu()
        del mgmodel

    epsilon = 1e-5 if mgargs.params_dtype == torch.float32 else 1e-2 ## 1e-5
    for idx, (hfh, mgh) in enumerate(zip(hf_hiddens, mg_hiddens)):
        assert len(hfh) == len(mgh)
        for k, hfv in hfh.items():
            mgv, hfv = mgh[k].cpu(), hfv.cpu()
            same_num = (hfv != mgv).sum()
            diff_num = ((hfv - mgv).abs() > epsilon).sum() # 
            diff_max = (hfv - mgv).abs().max()
            print(f'layer:{idx}, {k}, shape:{hfv.shape}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hfv.numel()}] diff_max:{diff_max}')

    if not is_oom:
        same_num = (hflogits != mglogits).sum()
        diff_num = ((hflogits - mglogits) > epsilon).sum()
        diff_max = (hflogits - mglogits).abs().max()
        print(f'logits: {same_num}, diff>{epsilon}:[{diff_num}/{hflogits.numel()}] diff_max:{diff_max}')

        hftokens = torch.argmax(hflogits, dim=-1)
        mgtokens = torch.argmax(mglogits, dim=-1)
        print(hftokens)
        print(mgtokens)
        diff_num = (hftokens != mgtokens).sum()
        print(f'tokens: {diff_num}/{hftokens.numel()}')


def add_extra_args(parser):
    # parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        mg_model = load_megatron_model_latest(args)
        if args.use_low_memory_convert:
          convert_checkpoint_from_megatron_to_transformers_low_memory(mg_model, args)
          if args.check_diff:
            hf_model = AutoModelForCausalLM.from_pretrained(args.save, trust_remote_code=True)
            if args.bf16:
              hf_model = hf_model.bfloat16()
            check_hf_mg_forward(hf_model, mg_model, args)
        else:
          config = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
          hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)
          if args.bf16:
            hf_model = hf_model.bfloat16()
          convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
          save_hfmodel(args, hf_model)
          if args.check_diff:
            check_hf_mg_forward(hf_model, mg_model, args)
        
    else:
        config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        # pdb.set_trace()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        # if args.q_lora_rank is None:
        check_hf_mg_forward(hf_model, mg_model, args)
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()
