# 进入ZJ-Megatron-LM目录
cd /xxx/ZJ-Megatron-LM
# ENV默认值：dsw 可选[dsw,dlc]
export ENV=dlc

################# 模型配置 ###############
# MODEL_SIZE默认值：200B,可选 [2B,16B,200B,600B]
export MODEL_SIZE=2B
# NUM_LAYERS默认值：9,28,60,61
export NUM_LAYERS=28
# AC默认值：none 可选 [full,sel,moe,offload,none] 推荐full
# QK_NOPE_HEAD_DIM默认值：128
export QK_NOPE_HEAD_DIM=128
# SEQ_LEN默认值：4096
export SEQ_LEN=4096
# PAD_LEN默认值：SEQ_LEN
# export PAD_LEN=

################# 训练配置 #################
# PP默认值：1 200B推荐6
export PP=2
# EP默认值：1 推荐 <=8
export EP=2
# MP_VP默认值：1
export MP_VP=1
export AC=full
# MP_AC_LAYERS 默认值：1
export MP_AC_LAYERS=1
# GLOBAL_BATCH_SIZE默认值：9600
export GLOBAL_BATCH_SIZE=400
# MICRO_BATCH_SIZE默认值：1
export MICRO_BATCH_SIZE=1
# DISPATCHER_TYPE默认值：alltoall_seq, 可选[alltoall_seq,alltoall,flex_deepep]
export DISPATCHER_TYPE=alltoall_seq
# FLASH_ATTENTION默认值：true 可选[true,false]
export FLASH_ATTENTION=true
# CKPT_FORMAT默认值：torch 可选[torch,torch_dist,torch_dist_async,torch_dist_no_optim] 推荐 torch_dist_async
export CKPT_FORMAT=torch_dist_async
# LR默认值：4E-5
export LR=4E-5
# MIN_LR默认值：4E-6
export MIN_LR=4E-6
# INIT_METHOD_STD默认值：0.006
export INIT_METHOD_STD=0.006

# MOE_ROUTER_GROUPS默认值：8
export MOE_ROUTER_GROUPS=8
# MOE_ROUTER_GROUPS_TOPK默认值：4
export MOE_ROUTER_GROUPS_TOPK=4

# ROUTER_SCORE_FUNC默认值：sigmod 可选[sigmod,softmax,pre_softmax]
export ROUTER_SCORE_FUNC=sigmod
# ROUTER_TOPK_SCALING_FACTOR默认值：none
export ROUTER_TOPK_SCALING_FACTOR=
# SAVE_INTERVAL默认值：1000
export SAVE_INTERVAL=1000
# SEQWARM默认值：off 配置--warmup-seq-length 0:2048,100:4096
# export SEQWARM=

# TP_PP_DP_MAP默认值：off  --use-tp-pp-dp-mapping
export TP_PP_DP_MAP=off

# TRAIN_ITERS默认值：TRAIN_TOKENS/GLOBAL_BATCH_SIZE/SEQ_LEN
# export TRAIN_ITERS=
# BIAS_MEAN默认值：FALSE , 是否 --moe-router-bias-mean-update-rate 1e-3 
# export BIAS_MEAN=
# CUSTOM_PIPE默认值：on 配置-no-custom-partition-with-smooth-weight
# export CUSTOM_PIPE=
# TRAIN_TOKENS默认值：2E+11
# export TRAIN_TOKENS=2E+11
# USE_FSDP默认值：FALSE
# export USE_FSDP=false
# WARMUP_TOKENS默认值：4194304000
# export WARMUP_TOKENS=
# OFFLOAD_OPTIMIZER默认值：FALSE  Optimizer精度优化和CPU Offloading 降低内存
# export OFFLOAD_OPTIMIZER=true
# PYTORCH_CUDA_ALLOC_CONF 是否使用内存碎片，降低内存
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


################## 相关路径和数据 #############
# MEGATRON_PATH：ZJ-Megatron-LM目录路径
export MEGATRON_PATH=
# TOKENIZER_PATH TOKENIZER路径
export TOKENIZER_PATH=
# OUTPUT_DIR 模型,log,tensorboard保持路径
export OUTPUT_DIR=
# PRETRAIN_CHECKPOINT_PATH 断点续训时需配置checkpoint路径
# export PRETRAIN_CHECKPOINT_PATH=
# DATASET_FILE数据集路径
export DATASET_FILE=
# VALID_DATASET_FILE 文件路径列表
export VALID_DATASET_FILE=

################# benchmark or profiling ##################
# PRINT_MFU默认值：true 打印MFU计算
export PRINT_MFU=true
# PROFILE默认值：off
export PROFILE=off
# 开启MFU benchmark mode 计算
export BENCHMARK_MFU=false


#################### 20250328新增  For MoE Stability############
# APPLY_NORM_HEAD默认不开启
export APPLY_NORM_HEAD=1
# WARMUP_ROUTER默认不开启
export WARMUP_ROUTER=5
# 开启MFU benchmark mode 指定iteration区间计算平均MFU
export BENCHMARK_MFU=false

bash examples/pretrain_021_dsv3.sh

