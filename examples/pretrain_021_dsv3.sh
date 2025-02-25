
ENV=${ENV:-dsw}

### BASE CONFIG ###
DEFAULT_MODEL_SIZE=A23B
MODEL_SIZE=${MODEL_SIZE:-${DEFAULT_MODEL_SIZE}}
# NUM_LAYERS=${NUM_LAYERS:-2}
BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
LR=4E-5
MIN_LR=4E-6
SEQ_LEN=${SEQ_LEN:-2048}
PAD_LEN=${PAD_LEN:-${SEQ_LEN}}
PR=bf16
# PR=fp8
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=1
PP=${PP:-1}
CP=1
EP=${EP:-1}
SP=false
DO=true
# DO=false
# FL=false
FL=true
SFT=false
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${AC:-none}
OPTIMIZER_OFFLOAD=false
SAVE_INTERVAL=1000
if [[ -z $DATASET_FILE ]] ; then
    echo "Missing environment variable DATASET_FILE."
    exit 1
fi
DATASET_PATH="$(cat ${DATASET_FILE})"
VALID_DATASET_PATH="$(cat ${VALID_DATASET_FILE})"
OUTPUT_DIR=${OUTPUT_DIR:-$PWD}
if [[ -z $TOKENIZER_PATH ]] ; then
    echo "Missing environment variable TOKENIZER_PATH."
    exit 1
fi
PRETRAIN_CHECKPOINT_PATH=${OUTPUT_DIR}/checkpoints

# the following two values will not be used when SFT is true
TRAIN_TOKENS=${TRAIN_TOKENS:-200000000000}
WARMUP_TOKENS=${WARMUP_TOKENS:-4194304000}
###############################

OUTPUT_BASEPATH=${OUTPUT_DIR}
### OTHERS ###

# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2
# export CUDNN_LOGERR_DBG=1
# export CUDNN_LOGDEST_DBG=stderr


### Begin of Script ###

set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
if [ -z $MEGATRON_PATH ]; then
    MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
fi
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

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

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
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

if [ $MODEL_SIZE = A23B ]; then

HIDDEN_SIZE=5120
NUM_ATTN_HEADS=128
NUM_LAYERS=${NUM_LAYERS:-64}
# NUM_LAYERS=60
# NUM_LAYERS=64
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=2048
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=128
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
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-grouped-gemm \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type aux_loss"
# --moe-grouped-gemm \
# --moe-router-topk-limited-devices 4 \

elif [ $MODEL_SIZE = A37B ]; then

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
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-grouped-gemm \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type aux_loss"
# --moe-grouped-gemm \
# --moe-router-topk-limited-devices 4 \

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

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
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
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=2000
# LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
PREFIX="pretrain-mcore-deepseek-v2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"

dataset_option=" \
    --data-path ${DATASET_PATH} \
    --data-cache-path ${OUTPUT_DIR}/data_cache \
    --split 989,10,1"

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
        --reset-position-ids \
        --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-pp-${PP}-ac-${AC}"
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
        --init-method-std 0.008 \
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
        --tokenizer-type 021Tokenizer \
        --tokenizer-model $TOKENIZER_PATH \
        --vocab-file $TOKENIZER_PATH/tokenizer.model \
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
        --ckpt-format torch \
        "
        # --num-workers 2 \
        # --patch-tokenizer-type DeepSeekV2Tokenizer \

tokenizer_options=" \
        --max-padding-length ${PAD_LEN} \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
	"

ckpt_options=" \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        "

# Turn on PyTorchProfiler in DSW
if [ $ENV = dsw ]; then
    export CUDA_LAUNCH_BLOCKING=1
    prof_options=" --profile --use-pytorch-profiler --profile-step-end 11 --profile-ranks 0 1 2 3 4 5 6 7 "
fi

# 开启pipeline_timer，将每个rank写到对应的文件中
if [ $RUN_PROFILE = on ]; then
    export PIPELINE_TIMER_LEVEL=2 
    export PIPELINE_TIMER_LOG_DIR=$OUTPUT_DIR/logs/${current_time}_${NNODES}/
    mkdir -p $PIPELINE_TIMER_LOG_DIR
fi

run_cmd="torchrun $DISTRIBUTED_ARGS ${MEGATRON_PATH}/pretrain_gpt.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${fl_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} ${packing_options} \
 ${uneven_split_option} ${prof_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
