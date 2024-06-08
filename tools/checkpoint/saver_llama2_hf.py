# From GitHub Gist: https://gist.github.com/devymex/734ff89ffb7ba047de177201ba90b3d1
import os, torch, torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, LlamaConfig

CHECK_EQUAL_WITH_HF = '' # A pretrain directory eg. '/data/models/llama-2-hf/7b-chat'

def add_arguments(parser):
    group = parser.add_argument_group(title='Llama-2 HF saver.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of megatron checkpoint')

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

    llama_conf = LlamaConfig(
        vocab_size              = mag_conf.padded_vocab_size,
        hidden_size             = mag_conf.hidden_size,
        intermediate_size       = mag_conf.ffn_hidden_size,
        num_hidden_layers       = mag_conf.encoder_num_layers,
        num_attention_heads     = mag_conf.num_attention_heads,
        num_key_value_heads     = mag_conf.num_query_groups,
        max_position_embeddings = mag_conf.max_position_embeddings,
        rms_norm_eps            = mag_conf.norm_epsilon,
        tie_word_embeddings     = not mag_conf.untie_embeddings_and_output_weights,
        attention_bias          = mag_conf.add_bias_linear,
        torch_dtype             = torch_dtype,
        model_type              = "llama",
        architectures           = ['LlamaForCausalLM'],
        transformers_version    = "4.33.1",
        )
    llama_conf.save_pretrained(args.save_dir)

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
        qkv_weight = message["qkv weight"]
        qkv_weight = qkv_weight.view(llama_conf.num_attention_heads, 3, -1, llama_conf.hidden_size)
        qkv_weight = qkv_weight.transpose(0, 1).reshape(3, llama_conf.hidden_size, llama_conf.hidden_size)
        set_hf_param(suffix + 'self_attn.q_proj', qkv_weight[0])
        set_hf_param(suffix + 'self_attn.k_proj', qkv_weight[1])
        set_hf_param(suffix + 'self_attn.v_proj', qkv_weight[2])
        set_hf_param(suffix + 'self_attn.o_proj', message["dense weight"])
        set_hf_param(suffix + 'mlp.down_proj', message["mlp l1 weight"])
    set_hf_param('model.norm', queue_get('final norm')['weight'])
    set_hf_param('lm_head', queue_get('output layer')['weight'])

    if CHECK_EQUAL_WITH_HF:
        print(f'Checking with given HF model {CHECK_EQUAL_WITH_HF}')
        ref_model = AutoModelForCausalLM.from_pretrained(CHECK_EQUAL_WITH_HF)
        ref_state_dict = ref_model.state_dict()
        assert sorted(list(ref_state_dict.keys())) == sorted(list(state_dict.keys()))
        for key in state_dict:
            assert torch.equal(ref_state_dict[key], state_dict[key])
        print(f'Check passed. {CHECK_EQUAL_WITH_HF} and {args.save_dir} are equal.')

    torch.save(state_dict, os.path.join(args.save_dir, 'pytorch_model.bin'))
