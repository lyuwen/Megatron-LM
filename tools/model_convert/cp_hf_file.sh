#hf_ckpt_path=/mnt/workspace/projects/superalignment/chengkun/pai-megatron/deepseek/deepseek-ckpts/deepseek_v3_3b_v2
hf_ckpt_path=/mnt/cpfs/users/mzy/moe/output/hf_test/init_hf_layer_pp_test
hf_ckpt_path=/mnt/cpfs/projects/superalignment/lufanfeng/deepseek-convert/data/deepseek_1745716657/output
hf_ckpt_path=/mnt/cpfs/users/mzy/moe/output/hf_test/16b_dsv2
output_path=/mnt/cpfs/users/mzy/moe/output/hf_test/16b_dsv2_2
mkdir -p ${output_path}
cp ${hf_ckpt_path}/*config.json ${output_path}/
# cp ${hf_ckpt_path}/tokenizer*.json ${output_path}/
cp ${hf_ckpt_path}/*.py ${output_path}/
cp ${hf_ckpt_path}/special_tokens_map.json ${output_path}/
cp ${hf_ckpt_path}/*tokenizer* ${output_path}/

