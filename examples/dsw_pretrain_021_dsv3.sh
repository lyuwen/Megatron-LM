cd ../
export ENV=dsw
export MODEL_SIZE=2B
export PP=2
export EP=2
export MP_VP=1
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=400
export AC=full
# export LR=4E-5
# export MIN_LR=4E-6
# export INIT_METHOD_STD=0.006
export SEQ_LEN=4096
# export PAD_LEN=4096
export FLASH_ATTENTION=true
export TOKENIZER_PATH=/mnt/workspace/users/liuzd/MoE/zjllm-llama3-tokenizer
# export WARMUP_TOKENS=
export NUM_LAYERS=28
export QK_NOPE_HEAD_DIM=128
# export MOE_ROUTER_GROUPS=8
# export MOE_ROUTER_GROUPS_TOPK=4
# export ROUTER_TOPK_SCALING_FACTOR=
# export BIAS_MEAN=
export DISPATCHER_TYPE=alltoall_seq
# export PROFILE=
# export SEQWARM=
# export CUSTOM_PIPE=
# export TP_PP_DP_MAP=
# export USE_FSDP=
# export OFFLOAD_OPTIMIZER= 
export MEGATRON_PATH=$PWD
export DATASET_FILE=/mnt/train-sample/train-sample.datalist.mini
export TRAIN_ITERS=1000
export SAVE_INTERVAL=10
export OUTPUT_DIR=/mnt/workspace/users/liuzd/MoE/run_deepseek/output-8c771980-20250327/code_test
export CKPT_FORMAT=torch_dist_no_optim
export PRETRAIN_CHECKPOINT_PATH=/mnt/workspace/users/liuzd/MoE/run_deepseek/output-8c771980-20250327/code_test/checkpoints/pretrain-zjmcore-dsv3-2B-lr-4E-5-minlr-4E-6-bs-1-gbs-400-seqlen-4096-pr-bf16-pp-2-ep-2-ac-full_20250328-1325
export PAO=true
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VALID_DATASET_FILE=/mnt/cpfs/code/validation/validation/valid-datalist

# 84bc91b-20250324
export ROUTER_SCORE_FUNC= pre_softmax

# 8c771980-20250327: For MoE Stability
export WARMUP_ROUTER=5
export APPLY_NORM_HEAD=1
bash examples/pretrain_021_dsv3.sh

