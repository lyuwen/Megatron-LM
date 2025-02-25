# From GitHub Gist: https://gist.github.com/devymex/734ff89ffb7ba047de177201ba90b3d1
import os, sys, torch, torch.multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer
from custom_models import LlamaMLAConfig, LlamaMLAForCausalLM

CHECK_EQUAL_WITH_HF = '' # A pretrain directory eg. '/data/models/llama-2-hf/7b-chat'

def add_arguments(parser):
    group = parser.add_argument_group(title='Llama-2 HF saver.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of megatron checkpoint')
    group.add_argument('--tokenizer-name-or-path', type=str, default=None,
                       help='Name or path of the tokenizer')

def save_checkpoint(queue: mp.Queue, args):
    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    md = queue_get()

    # Verify compatibility of args
    assert hasattr(md, 'checkpoint_args')
    assert md.model_type == 'GPT'
    assert md.multi_latent_attention
    mag_conf = md.checkpoint_args
    torch_dtype = torch.float32
    if mag_conf.bf16:
        assert mag_conf.fp16 == False
        torch_dtype = torch.bfloat16
    elif mag_conf.fp16:
        assert mag_conf.bf16 == False
        torch_dtype = torch.float16
    assert mag_conf.swiglu == True
    assert mag_conf.rotary_percent == 1.0

    tokenizer = None
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    if args.tokenizer_name_or_path is None and os.path.isdir(mag_conf.tokenizer_model):
        args.tokenizer_name_or_path = mag_conf.tokenizer_model
    if args.tokenizer_name_or_path is not None:
        print(f"Loading tokenizer from {args.tokenizer_name_or_path}.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        bos_token_id = getattr(tokenizer, "bos_token_id", bos_token_id)
        eos_token_id = getattr(tokenizer, "eos_token_id", eos_token_id)
        pad_token_id = getattr(tokenizer, "pad_token_id", pad_token_id)

    llama_conf = LlamaMLAConfig(
        vocab_size              = mag_conf.padded_vocab_size,
        hidden_size             = mag_conf.hidden_size,
        intermediate_size       = mag_conf.ffn_hidden_size,
        num_hidden_layers       = mag_conf.encoder_num_layers,
        num_attention_heads     = mag_conf.num_attention_heads,
        num_key_value_heads     = mag_conf.num_query_groups,
        #
        kv_lora_rank            = mag_conf.kv_lora_rank,
        q_lora_rank             = mag_conf.q_lora_rank,
        qk_head_dim             = mag_conf.qk_head_dim,
        qk_pos_emb_head_dim     = mag_conf.qk_pos_emb_head_dim,
        v_head_dim              = mag_conf.v_head_dim,
        rotary_scaling_factor   = mag_conf.rotary_scaling_factor,
        #
        max_position_embeddings = mag_conf.max_position_embeddings,
        rms_norm_eps            = mag_conf.norm_epsilon,
        bos_token_id            = bos_token_id,
        eos_token_id            = eos_token_id,
        pad_token_id            = pad_token_id,
        tie_word_embeddings     = not mag_conf.untie_embeddings_and_output_weights,
        rope_theta              = mag_conf.rotary_base,
        attention_bias          = mag_conf.add_bias_linear,
        torch_dtype             = torch_dtype,
        model_type              = "llamamla",
        architectures           = ['LlamaMLAForCausalLM'],
        transformers_version    = "4.33.1",
        auto_map                = {'AutoConfig': 'configuration_llamamla.LlamaMLAConfig',
                                   'AutoModelForCausalLM': 'modeling_llamamla.LlamaMLAForCausalLM'},
        )
    llama_conf.save_pretrained(args.save_dir)
    if tokenizer is not None:
        print(f"Save tokenizer to {args.save_dir}.")
        tokenizer.save_pretrained(args.save_dir)

    state_dict = {}
    def set_hf_param(name, tensor: torch.Tensor):
        weight_name = f'{name}.weight'
        state_dict[weight_name] = tensor

    set_hf_param('model.embed_tokens', queue_get("embeddings")["word embeddings"])
    for i_layer in range(llama_conf.num_hidden_layers):
        message = queue_get(f"transformer layer {i_layer}")
        suffix = f'model.layers.{i_layer}.'
        set_hf_param(suffix + 'input_layernorm', message["input norm weight"])
        set_hf_param(suffix + 'post_attention_layernorm', message["post norm weight"])
        set_hf_param(suffix + 'mlp.gate_proj', message["mlp l0 weight W"])
        set_hf_param(suffix + 'mlp.up_proj', message["mlp l0 weight V"])
        # qkv_weight = message["qkv weight"]
        # qkv_weight = qkv_weight.view(llama_conf.num_attention_heads, 3, -1, llama_conf.hidden_size)
        # qkv_weight = qkv_weight.transpose(0, 1).reshape(3, llama_conf.hidden_size, llama_conf.hidden_size)
        #
        if mag_conf.q_lora_rank is None:
            set_hf_param(suffix + 'self_attn.q_proj', message["q_proj_weight"])
        else:
            set_hf_param(suffix + 'self_attn.q_a_proj', message["q_down_proj_weight"])
            set_hf_param(suffix + 'self_attn.q_b_proj', message["q_up_proj_weight"])
            set_hf_param(suffix + 'self_attn.q_a_layernorm', message["q_layernorm_weight"])
        set_hf_param(suffix + 'self_attn.kv_a_proj', message["kv_down_proj_weight"])
        set_hf_param(suffix + 'self_attn.kv_b_proj', message["kv_up_proj_weight"])
        set_hf_param(suffix + 'self_attn.kv_a_layernorm', message["kv_layernorm_weight"])
        set_hf_param(suffix + 'self_attn.o_proj', message["dense weight"])
        #
        set_hf_param(suffix + 'mlp.down_proj', message["mlp l1 weight"])
    set_hf_param('model.norm', queue_get('final norm')['weight'])
    set_hf_param('lm_head', queue_get('output layer')['weight'])

    #  if CHECK_EQUAL_WITH_HF:
    #      print(f'Checking with given HF model {CHECK_EQUAL_WITH_HF}')
    #      ref_model = AutoModelForCausalLM.from_pretrained(CHECK_EQUAL_WITH_HF)
    #      ref_state_dict = ref_model.state_dict()
    #      assert sorted(list(ref_state_dict.keys())) == sorted(list(state_dict.keys()))
    #      for key in state_dict:
    #          assert torch.equal(ref_state_dict[key], state_dict[key])
    #      print(f'Check passed. {CHECK_EQUAL_WITH_HF} and {args.save_dir} are equal.')

    torch.save(state_dict, os.path.join(args.save_dir, 'pytorch_model.bin'))
