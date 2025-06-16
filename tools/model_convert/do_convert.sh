cd /mnt/workspace/users/mzy/moe/ZJ-Megatron-LM-0.0.11b1-20250328_3/tools/convert
model_size=DSnew2.5B-baseline
model_size=DSnew16B_v2_20250409
tp=1
pp=1
first_layers_pp=0
last_layers_pp=0
vp=0
ep=4
hf_ckpt_path=/tmp
load=/mnt/cpfs/projects/superalignment/zdd/output/ZJ-Megatron-SFT/MoE-3B-16B-20250422-lufanfeng-128-3-2t/checkpoints/finetune-zjmcore-dsv2-16BV2-lr-2E-5-minlr-1E-6-bs-4-gbs-128-seqlen-4096_torch-pr-bf16-pp-1-ac-full-noaux
iteration=022154
save=/mnt/cpfs/users/mzy/moe/output/hf_test/16b_dsv2_2
check_diff=true
low_memory_convert=true
save_num_files=3
logfile=/mnt/cpfs/users/mzy/moe/convert.info
bash hf2mcore_deepseek_v2_moe_convertor_llama3_tokenizer_v3.sh \
${model_size} \
${load} \
${save} \
${tp} \
${pp} \
${first_layers_pp} \
${last_layers_pp} \
${vp} \
${ep} \
bf16 \
true \
${hf_ckpt_path} \
${iteration} \
${check_diff} \
${low_memory_convert} \
${save_num_files} \
2>&1 | tee ${logfile}
