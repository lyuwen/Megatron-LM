set -ex

# ENV=${ENV:-dsw}
ENV=dlc

### BASE CONFIG ###
DEFAULT_MODEL_SIZE=16BV2 #16BV2
MODEL_SIZE=${MODEL_SIZE:-${DEFAULT_MODEL_SIZE}}
BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
DEFAULT_LR=3E-5 #4E-4
LR=${LR:-${DEFAULT_LR}}
DEFAULT_MIN_LR=0
MIN_LR=${MIN_LR:-${DEFAULT_MIN_LR}}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.006} # 0.006 

SEQ_LEN=${SEQ_LEN:-4096}
PAD_LEN=${PAD_LEN:-${SEQ_LEN}}
PR=bf16
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=1
PP=2 #2
CP=1
EP=16
SP=false
DO=true
FL=true
SFT=false
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${AC:-none} # full
# OPTIMIZER_OFFLOAD=false to be checked
SAVE_INTERVAL=3720
OUTPUT_DIR=${OUTPUT_DIR:-$PWD}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/data/lufanfeng/Megatron-Core/zjllm-llama3-tokenizer/"}
if [ -z $TOKENIZER_PATH ] ; then
    echo "Missing environment variable TOKENIZER_PATH."
    exit 1
fi
if [ ${ASYNC_SAVE:-false} = true ] ; then
    ckpt_options=" --ckpt-format torch_dist --async-save --auto-detect-ckpt-format"
else
    ckpt_options=" --ckpt-format ${CKPT_FORMAT:-torch} "
fi

PRETRAIN_CHECKPOINT_PATH="/mnt/data/gft/output/MoE-3B-16B-20250325-8b1-spring-lufanfeng/checkpoints/pretrain-mcore-deepseek-v2-DS16B-lr-4.2e-4-minlr-4.2e-5-bs-1-gbs-4800-seqlen-4096-pr-bf16-tp-1-pp-2-cp-1-ac-false-do-true-sp-true-ti-381469-wi-1920"
# PRETRAIN_CHECKPOINT_PATH="/mnt/data/gft/output/MoE-3B-16B-20250325-8b1-spring-lufanfeng/checkpoints/pretrain-mcore-deepseek-v2-DS16B-lr-4.2e-4-minlr-4.2e-5-bs-1-gbs-4800-seqlen-4096-pr-bf16-tp-1-pp-2-cp-1-ac-false-do-true-sp-true-ti-381469-wi-1920/pretrain-mcore-deepseek-v2-DS16B-lr-4.2e-4-minlr-4.2e-5-bs-1-gbs-4800-seqlen-4096-pr-bf16-tp-1-pp-2-cp-1-ac-false-do-true-sp-true-ti-381469-wi-1920_converted/torch_dist"
TRAIN_ITERS=12000
LR_WARMUP_ITERS=1200
# training configuraitons
if [ -z $TRAIN_ITERS ] ; then
    echo "Missing environment variable TRAIN_ITERS."
    exit 1
fi
TRAIN_TOKENS=${TRAIN_TOKENS:-200000000000}
WARMUP_TOKENS=${WARMUP_TOKENS:-4194304000}
TOTAL_TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
TRAIN_ITERS=${TRAIN_ITERS:-${TOTAL_TRAIN_ITERS}}
OUTPUT_DIR="/mnt/data/zdd/output/ZJ-Megatron-SFT/MoE-3B-16B-20250325-8b1-spring-lufanfeng"
OUTPUT_BASEPATH=${OUTPUT_DIR}
### OTHERS ###
if [ ${DEBUG} = on ] ; then
    export NVTE_DEBUG=1
    export NVTE_DEBUG_LEVEL=2
    export CUDNN_LOGERR_DBG=1
    export CUDNN_LOGDEST_DBG=stderr
fi

### Begin of Script ###
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
if [ -z $MEGATRON_PATH ]; then
    MEGATRON_PATH=$( dirname ${CURRENT_DIR})
fi
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ $ENV = dsw ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=8
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

MP_VP=2 #2
if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi


TIMESTAMP=$(date "+%Y%m%d-%H%M")
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
    --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    --tee 3 --log_dir ${OUTPUT_DIR}/logs/${TIMESTAMP}"

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    fl_options=" --attention-backend flash "
elif [ $FL = false ]; then
    # export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    fl_options=" --attention-backend unfused "
fi

if [ $MODEL_SIZE = 2B ]; then

HIDDEN_SIZE=1280
NUM_ATTN_HEADS=10
NUM_LAYERS=${NUM_LAYERS:-9}
INTERMEDIATE_SIZE=8192
MOE_INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=256
# Q_LORA_RANK=1536 # 后训练组删除
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=7
NUM_SHARED_EXPERTS=1
MOE_LAYER_FREQ=1
MOE_FIRST_K_DENSE_REPLACE=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.000 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-grouped-gemm \
    --moe-router-num-groups ${MOE_ROUTER_GROUPS} \
    --moe-router-group-topk ${MOE_ROUTER_GROUPS_TOPK} \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-bias-update-rate 0.000"
    # --q-lora-rank ${Q_LORA_RANK}"
    if [ ${ROUTER_TOPK_SCALING_FACTOR:-none} != none ]; then
    moe_options=" ${moe_options}  --moe-router-topk-scaling-factor ${ROUTER_TOPK_SCALING_FACTOR} "
    fi

elif [ $MODEL_SIZE = 16B ]; then

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=${NUM_LAYERS:-28}
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=256
# Q_LORA_RANK=1536 # 后训练组删除
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6
MOE_FIRST_K_DENSE_REPLACE=1

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-grouped-gemm \
    --moe-router-num-groups ${MOE_ROUTER_GROUPS} \
    --moe-router-group-topk ${MOE_ROUTER_GROUPS_TOPK} \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-bias-update-rate 1e-3"
    # --q-lora-rank ${Q_LORA_RANK} \
    if [ ${ROUTER_TOPK_SCALING_FACTOR:-none} != none ]; then
    moe_options=" ${moe_options} --moe-router-topk-scaling-factor ${ROUTER_TOPK_SCALING_FACTOR} "
    fi

elif [ $MODEL_SIZE = 16BV2 ]; then

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=${NUM_LAYERS:-28}
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=256
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6
MOE_AUX_LOSS_COEFF=0 #1e-3
MOE_FIRST_K_DENSE_REPLACE=0

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss \
    --moe-token-dispatcher-type allgather \
    --moe-router-score-function softmax \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 1.0
    "

elif [ $MODEL_SIZE = 200B ]; then

HIDDEN_SIZE=5120
NUM_ATTN_HEADS=128
NUM_LAYERS=${NUM_LAYERS:-60} 
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=160
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
MOE_FIRST_K_DENSE_REPLACE=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-grouped-gemm \
    --moe-router-num-groups ${MOE_ROUTER_GROUPS} \
    --moe-router-group-topk ${MOE_ROUTER_GROUPS_TOPK} \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-bias-update-rate 1e-3"
# --moe-token-drop-policy probs 
# --moe-grouped-gemm \
# --moe-router-topk-limited-devices 4 \
    if [ ${ROUTER_TOPK_SCALING_FACTOR:-none} != none ]; then
    moe_options=" ${moe_options} --moe-router-topk-scaling-factor ${ROUTER_TOPK_SCALING_FACTOR} "
    fi

elif [ $MODEL_SIZE = 600B ]; then

HIDDEN_SIZE=7168
NUM_ATTENTION_HEADS=128
NUM_LAYERS=${NUM_LAYERS:-61}
INTERMEDIATE_SIZE=18432
MOE_INTERMEDIATE_SIZE=2048
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=467
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=256
ROUTER_TOPK=8
NUM_SHARED_EXPERTS=1
MOE_LAYER_FREQ=1
MOE_FIRST_K_DENSE_REPLACE=3
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-grouped-gemm \
    --moe-router-num-groups ${MOE_ROUTER_GROUPS} \
    --moe-router-group-topk ${MOE_ROUTER_GROUPS_TOPK} \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-bias-update-rate 1e-3"
    if [ ${ROUTER_TOPK_SCALING_FACTOR:-none} != none ]; then
    moe_options=" ${moe_options} --moe-router-topk-scaling-factor ${ROUTER_TOPK_SCALING_FACTOR} "
    fi

else

echo "Unsupported model size: ${MODEL_SIZE}"
exit 1

fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"
 

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = moe ]; then
    activation_checkpoint_options=" \
        --moe-layer-recompute \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
# for custom AC
elif [ $AC = custom ]; then
    #TODO: fill in custom AC options
    activation_checkpoint_options=" \
    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

te_options=" \
        --transformer-impl transformer_engine"

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    uneven_split_option=" \
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH \
            --override-opt_param-scheduler \
            --no-load-optim \
            --no-load-rng \
            --reset-iterations \
            --reset-dataloader \
            "
                        # --reset-dataloader \
fi

# TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-0}
# LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=${LR_DECAY_ITERS:-${TRAIN_ITERS}}
PREFIX="finetune-zjmcore-dsv2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}_torch"

# TRAIN_DATA="'/mnt/data/lusongshuo/datasets/tulu3_mixed_part_1.json' '/mnt/data/lusongshuo/datasets/tulu3_mixed_part_2.json' '/mnt/data/lusongshuo/datasets/tulu3_mixed_part_3.json' '/mnt/data/lusongshuo/datasets/tulu3_mixed_part_4.json' '/mnt/data/lusongshuo/datasets/tulu3_mixed_part_5.json'"

# TRAIN_DATA="/mnt/data/lusongshuo/datasets/tulu3_mixed_test.json"

TRAIN_DATA="/mnt/data/lusongshuo/ZJ-Megatron-LM/megatron/core/datasets/tuluv3_train"

# 
dataset_option=" \
    --train-data-path $TRAIN_DATA \
    --data-cache-path ${OUTPUT_DIR}/data_cache \
    --max-padding-length ${PAD_LEN} \
    --dataloader-type cyclic \
    "
    # --valid-data-path ${VALID_DATA:-/mnt/data/llf/alpaca_gpt4_data_en.json} \
    # --test-data-path ${TEST_DATA:-/mnt/data/llf/alpaca_gpt4_data_en.json} \
##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-pp-${PP}-ac-${AC}-noaux"
# NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/data_cache/"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoints/"
mkdir -p "${OUTPUT_BASEPATH}/logs/"
current_time=$(date "+%Y.%m.%d-%H.%M")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoints/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
# find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
#find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std ${INIT_METHOD_STD} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model $TOKENIZER_PATH \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --multi-latent-attention \
        --distributed-timeout-minutes 40"
        # --num-workers 2 \
        # --patch-tokenizer-type DeepSeekV2Tokenizer \

        # tokenizer_options=" \
        #         --max-padding-length ${PAD_LEN} \
        #         --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        #         "

# ckpt_options=" \
#         --no-load-optim \
#         --no-load-rng \
#         --no-save-optim \
#         "

# Turn on PyTorchProfiler in DSW
# if [ $ENV = dsw ]; then
#     export CUDA_LAUNCH_BLOCKING=1
#     prof_options=" --profile --use-pytorch-profiler --profile-step-end 11 --profile-ranks 0 1 2 3 4 5 6 7 "
# fi

# 开启pipeline_timer，将每个rank写到对应的文件中
if [ ${PROFILE:-off} = on ]; then
    export PIPELINE_TIMER_LEVEL=3
    export PIPELINE_TIMER_LOG_DIR=$OUTPUT_DIR/logs/${current_time}_${NNODES}/
    mkdir -p $PIPELINE_TIMER_LOG_DIR
fi

if [ ${SEQWARM:-off} = on ]; then
seqwarm_options=" --warmup-seq-length 0:2048,100:4096 "
fi

# new_options=" --checkpoint-kv-up-proj --recompute-inputlayer-rmsnorm --recompute-pre-mlp-rmsnorm "
if [ ${CUSTOM_PIPE:-on} = off ]; then
    new_options=" ${new_options} --no-custom-partition-with-smooth-weight "
fi

# Use TP-PP-DP mapping
if [ ${TP_PP_DP_MAP:-off} = on ] ; then
    new_options=" ${new_options} --use-tp-pp-dp-mapping "
fi

# User custom FSDP from Megatron Core
if [ ${USE_FSDP:-false} = true ] ; then
    fsdp_options="\
        --use-custom-fsdp \
        --data-parallel-sharding-strategy optim_grads_params \
        --no-gradient-accumulation-fusion \
        --calculate-per-token-loss \
        "
    unset CUDA_MAX_CONNECTIONS
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi

# User Optimizer CPU Offloading
if [ ${OFFLOAD_OPTIMIZER:-false} = true ] ; then
    new_options=" ${new_options} --optimizer-cpu-offload --use-precision-aware-optimizer \
        --main-grads-dtype bf16 "
fi

run_cmd="torchrun $DISTRIBUTED_ARGS ${MEGATRON_PATH}/finetune_gpt.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${fl_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} \
 ${uneven_split_option} ${prof_options} ${seqwarm_options} ${new_options} ${fsdp_options} ${ckpt_options}"

echo ${run_cmd}
[[ $RANK = 0 ]] && echo echo ${run_cmd} > ${OUTPUT_DIR}/logs/${TIMESTAMP}/cmd.sh
eval ${run_cmd}

