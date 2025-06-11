set -eo pipefail
set -x  # 如果你要调试的话
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV=${ENV:-dsw}

### BASE CONFIG ###
DEFAULT_MODEL_SIZE=200B
MODEL_SIZE=${MODEL_SIZE:-${DEFAULT_MODEL_SIZE}}
BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-9600}
DEFAULT_LR=2.4E-4
LR=${LR:-${DEFAULT_LR}}
DEFAULT_MIN_LR=2.4E-5
MIN_LR=${MIN_LR:-${DEFAULT_MIN_LR}}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.006} # 0.006 

SEQ_LEN=${SEQ_LEN:-4096}
PAD_LEN=${PAD_LEN:-${SEQ_LEN}}
PR=${PR:-bf16}
### BASE CONFIG ###


### PARALLEL / BOOL OPTION ###
PP=${PP:-1} # 6
EP=${EP:-1} # 8
FL=${FLASH_ATTENTION:-true} # true
TP=1
CP=1
SP=false
DO=true
SFT=false
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${AC:-none} # full
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
if [[ -z $DATASET_FILE ]] ; then
    echo "Missing environment variable DATASET_FILE."
    exit 1
fi
DATASET_PATH="$(cat ${DATASET_FILE})"
#VALID_DATASET_PATH="$(cat ${VALID_DATASET_FILE})"
VALID_DATASET_PATH=${DATASET_PATH}
DEFUALT_DATASET_CACHE_PATH=${OUTPUT_DIR}/data_cache
DATASET_CACHE_PATH=${DATASET_CACHE_PATH:-$DEFUALT_DATASET_CACHE_PATH}

OUTPUT_DIR=${OUTPUT_DIR:-$PWD}
if [[ -z $TOKENIZER_PATH ]] ; then
    echo "Missing environment variable TOKENIZER_PATH."
    exit 1
fi
CKPT_FORMAT=${CKPT_FORMAT:-torch}
if [ ${CKPT_FORMAT} = torch_dist_async ] ; then
    ckpt_options=" --ckpt-format torch_dist --async-save "
elif [ ${CKPT_FORMAT} = torch_dist_no_optim ] ; then
    ckpt_options=" --ckpt-format torch_dist --no-save-optim --no-load-optim"
elif [ ${CKPT_FORMAT} = torch_dist ] ; then
    ckpt_options=" --ckpt-format torch_dist "
elif [ ${CKPT_FORMAT} = torch ] ; then
    ckpt_options=" --ckpt-format torch "
elif [ ${CKPT_FORMAT} = torch_no_optim ] ; then
    ckpt_options=" --ckpt-format torch --no-save-optim --no-load-optim"
fi
ckpt_options="${ckpt_options} \
            --dist-ckpt-strictness log_all \
"
            #--auto-detect-ckpt-format \
            #--no-ckpt-fully-parallel-save \
            #--ckpt-assume-constant-structure \
            #--use-checkpoint-opt_param-scheduler"
            #--exit-on-missing-checkpoint \

# training configuraitons
TRAIN_TOKENS=${TRAIN_TOKENS:-11692571197196}
WARMUP_TOKENS=${WARMUP_TOKENS:-19660800000}
TOTAL_TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
TRAIN_ITERS=${TRAIN_ITERS:-${TOTAL_TRAIN_ITERS}}

OUTPUT_BASEPATH=${OUTPUT_DIR}
### OTHERS ###
if [[ ${DEBUG} = on ]]; then
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
export NVTE_BIAS_GELU_NVFUSION=0

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
    if [[ ${MP_VP_RANK:-none} != none ]]; then
        vp_options=" \
            --num-virtual-stages-per-pipeline-rank ${MP_VP_RANK}"
    else
        vp_options=""
    fi
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    fl_options=" --attention-backend flash "
elif [ $FL = false ]; then
    # export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    fl_options=" --attention-backend unfused "
fi

if [ $MODEL_SIZE = 2B ]; then

HIDDEN_SIZE=${HIDDEN_SIZE:-1280}
NUM_ATTN_HEADS=10
NUM_LAYERS=${NUM_LAYERS:-9}
INTERMEDIATE_SIZE=8192
MOE_INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${SEQ_LEN}}
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
MOE_FIRST_K_DENSE_REPLACE=2
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
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-grouped-gemm"


elif [ $MODEL_SIZE = 16B ]; then

HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
NUM_ATTN_HEADS=16
NUM_LAYERS=${NUM_LAYERS:-28}
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${SEQ_LEN}}
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
    --moe-grouped-gemm"

elif [ $MODEL_SIZE = 200B ]; then

HIDDEN_SIZE=${HIDDEN_SIZE:-5120}
NUM_ATTN_HEADS=128
NUM_LAYERS=${NUM_LAYERS:-60} 
INTERMEDIATE_SIZE=${INTERMEDIATE_SIZE:-12288}
MOE_INTERMEDIATE_SIZE=${MOE_INTERMEDIATE_SIZE:-1536}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${SEQ_LEN}}
#MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=${QK_NOPE_HEAD_DIM:-128} 
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=${SCALE_FACTOR:-40}
NUM_EXPERTS=${NUM_EXPERTS:-160}
ROUTER_TOPK=${ROUTER_TOPK:-6}
NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-2}
MOE_LAYER_FREQ=1
MOE_FIRST_K_DENSE_REPLACE=${MOE_FIRST_K_DENSE_REPLACE:-1}
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
    --moe-grouped-gemm"

elif [ $MODEL_SIZE = 600B ]; then

HIDDEN_SIZE=7168
NUM_ATTN_HEADS=128
NUM_LAYERS=${NUM_LAYERS:-61}
INTERMEDIATE_SIZE=18432
MOE_INTERMEDIATE_SIZE=2048
MAX_POSITION_EMBEDDINGS=163840
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
    --moe-grouped-gemm"

elif [ $MODEL_SIZE = 1000B ]; then

HIDDEN_SIZE=${HIDDEN_SIZE:-8192}
NUM_ATTN_HEADS=128
NUM_LAYERS=${NUM_LAYERS:-59} 
INTERMEDIATE_SIZE=${INTERMEDIATE_SIZE:-18432}
MOE_INTERMEDIATE_SIZE=${MOE_INTERMEDIATE_SIZE:-2048}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-163840}
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=${QK_NOPE_HEAD_DIM:-128} 
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=${SCALE_FACTOR:-40}
NUM_EXPERTS=${NUM_EXPERTS:-384}
ROUTER_TOPK=${ROUTER_TOPK:-8}
NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-1}
MOE_LAYER_FREQ=1
MOE_FIRST_K_DENSE_REPLACE=${MOE_FIRST_K_DENSE_REPLACE:-3}
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-z-loss-coeff 0.0001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-grouped-gemm"
else

echo "Unsupported model size: ${MODEL_SIZE}"
exit 1

fi

LOAD_BALANCE_TYPE=${LOAD_BALANCE_TYPE:-aux_loss}
if [ ${LOAD_BALANCE_TYPE} = aux_loss ] ; then
    moe_options=" ${moe_options} --moe-router-load-balancing-type ${LOAD_BALANCE_TYPE} "
elif [ ${LOAD_BALANCE_TYPE} = seq_aux_loss ] ; then
    moe_options=" ${moe_options} --moe-router-load-balancing-type ${LOAD_BALANCE_TYPE} "
else
echo "Unsupported moe-router-load-balancing-type: ${LOAD_BALANCE_TYPE}"
exit 1
fi

MOE_ROUTER_GROUPS=${MOE_ROUTER_GROUPS:-0} # 8
MOE_ROUTER_GROUPS_TOPK=${MOE_ROUTER_GROUPS_TOPK:-0} # 3

if [ $MOE_ROUTER_GROUPS -gt 0 ] && [ $MOE_ROUTER_GROUPS_TOPK -gt 0 ]; then
    moe_options=" ${moe_options}  --moe-router-num-groups ${MOE_ROUTER_GROUPS} \
    --moe-router-group-topk ${MOE_ROUTER_GROUPS_TOPK} "
fi

if [[ ${ROUTER_TOPK_SCALING_FACTOR:-none} != none ]]; then
moe_options=" ${moe_options} --moe-router-topk-scaling-factor ${ROUTER_TOPK_SCALING_FACTOR} "
fi

if [[ ${ROUTER_BIAS:-false} = true ]]; then
moe_options=" ${moe_options} --moe-router-enable-expert-bias \
            --moe-router-bias-update-rate 1e-3"
fi

if [[ ${BIAS_MEAN:-false} = true ]]; then
moe_options=" ${moe_options} --moe-router-bias-mean-update-rate 1e-3 "
fi

DISPATCHER_TYPE=${DISPATCHER_TYPE:-alltoall_seq}
if [ $DISPATCHER_TYPE = alltoall_seq ]; then
    moe_options=" ${moe_options}  --moe-token-dispatcher-type alltoall_seq  "
elif [ $DISPATCHER_TYPE = alltoall ]; then
    moe_options=" ${moe_options}  --moe-token-dispatcher-type alltoall --moe-shared-expert-overlap "
elif [ $DISPATCHER_TYPE = flex_deepep ]; then
    moe_options=" ${moe_options} --moe-token-dispatcher-type flex --moe-enable-deepep --moe-shared-expert-overlap "
else
echo "Unsupported dispatcher type: ${DISPATCHER_TYPE}"
exit 1
fi

ROUTER_SCORE_FUNC=${ROUTER_SCORE_FUNC:-pre_softmax}
if [ $ROUTER_SCORE_FUNC = sigmod ]; then
    moe_options=" ${moe_options}  --moe-router-score-function sigmoid \
                                    --moe-topk-router-fusion \
                                    "
elif [ $ROUTER_SCORE_FUNC = softmax ]; then
    moe_options=" ${moe_options}  --moe-router-score-function softmax "
elif [ $ROUTER_SCORE_FUNC = pre_softmax ]; then
    moe_options=" ${moe_options} --moe-router-score-function softmax --moe-router-pre-softmax "
else
echo "Unsupported router score function: ${ROUTER_SCORE_FUNC}"
exit 1
fi

# For MoE Stability
if [[ ${WARMUP_ROUTER:-0} -gt 0 ]]; then
    moe_options=" ${moe_options}  --moe-warmup-router  ${WARMUP_ROUTER}  "
fi

if [ ! -z ${APPLY_NORM_HEAD} ];then
    moe_options=" ${moe_options}  --moe-apply-norm-head "
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather \
"
#    --no-align-param-gather \
#    --no-scatter-gather-tensors-in-pipeline \

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option=" ${comm_overlap_option} \
        --tp-comm-overlap \
    "
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
        --recompute-granularity selective \
        --recompute-modules ${RECOMPUTE_MODULES:-"core_attn moe_act layernorm mla_up_proj mlp moe"} \
    "
    if [[ ${MOE_PERMUTE_CHECKPOINT:-none} != none ]]; then
        activation_checkpoint_options=" ${activation_checkpoint_options} \
            --moe-perm-checkpoint ${MOE_PERMUTE_CHECKPOINT} 
        "
    fi
elif [ $AC = permckpt ]; then
    activation_checkpoint_options=" \
        --recompute-granularity selective \
        --recompute-beside-moe \
        --recompute-modules moe \
        --moe-perm-checkpoint ${MOE_PERMUTE_CHECKPOINT:-half} \
    "
elif [ $AC = moeckpt ]; then
    activation_checkpoint_options=" \
        --recompute-beside-moe \
    "
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
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
    #镜像：te23-glfp8-baseline-0424
    #   TE: 2.3_dev
    #   OpenMixOpl: 0.1.15.2
    #   OpenMixOpl_GroupGemm: 1.2.0

    # use blockwise recipe in gl, delayed/tensorwise recipe in others
   pr_options=" \
        --bf16 \
        --fp8-format hybrid  \
        --fp8-recipe delayed \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --v3-fp8-grouped-linear \
    "
        #--v3-fp8-linear \
elif [ $PR = fp8_delayed ]; then
    # normal fp8, support delayed/tensorwise recipe in all
    pr_options=" \
         --bf16 \
         --fp8-format hybrid \
         --fp8-recipe delayed \
         --fp8-amax-compute-algo max \
         --fp8-amax-history-len 1024 \
     "
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

uneven_split_option=""
if [[ ${MP_PP0_LAYERS:-0} -gt 0 ]]; then
    uneven_split_option="${uneven_split_option} \
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
fi
if [[ ${MP_PPN_LAYERS:-0} -gt 0 ]]; then
    uneven_split_option="${uneven_split_option} \
        --decoder-last-pipeline-num-layers ${MP_PPN_LAYERS}
    "
fi

if [ -z ${DLC_JOB_ID} ]; then
    DLC_JOB_ID=${HOSTNAME:0:19}
fi

LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
PREFIX="pretrain-dsv3-${MODEL_SIZE}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}"
NAME="${PREFIX}-pp-${PP}-ep-${EP}-ac-${AC}_${DLC_JOB_ID}"
TIMESTAMP=$(date "+%Y%m%d-%H%M")
# NAME_JOBID_TIME="${PREFIX}-pp-${PP}-ep-${EP}-ac-${AC}_${DLC_JOB_ID}_${TIMESTAMP}"

PRETRAIN_CHECKPOINT_PATH_DEFAULT="${OUTPUT_BASEPATH}/checkpoints/${NAME}"
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-$PRETRAIN_CHECKPOINT_PATH_DEFAULT}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoints/${NAME}"
mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
# 重启时，如果保存目录中已存在ckpt,则断点续训将从该目录继续，不再从原ckpt目录加载重新训练 0417 lzd
if [[ -f ${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt ]]; then
    PRETRAIN_CHECKPOINT_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH}
fi
# find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
#find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

# DEBUG model without saving optim
if [[ ${DEBUG_PRETRAIN_CHECKPOINT_PATH:-none} != none ]]; then
    PRETRAIN_CHECKPOINT_PATH=$DEBUG_PRETRAIN_CHECKPOINT_PATH
    ckpt_options=" ${ckpt_options} \
        --auto-detect-ckpt-format \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        --no-save-rng \
        "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH "
fi

NUM_WORKERS=${NUM_WORKERS:-1}
dataset_option=" \
    --data-path ${DATASET_PATH} \
    --data-cache-path ${DATASET_CACHE_PATH} \
    --num-workers ${NUM_WORKERS} \
    --distributed-timeout-minutes 60 \
    --split 100,0,0"

if [[ ${MOCK_DATASET:-false} = true ]]; then
    dataset_option=" \
        --mock-data \
    "
fi

if [[ ${NO_MMAP_BIN_FILES:-false} = true ]]; then
    dataset_option=" ${dataset_option} --no-mmap-bin-files "
fi

# 日志目录中加入worker_id,快速定位相关日志 0417 lzd
if [[-z ${RANK} ]]; then
    WORKER_ID="UNKNOW_RANK"
else
    WORKER_ID=worker_${RANK}
    # WORK_ID=${ATLAS_POD_NAME}+${RANK}
fi

mkdir -p ${OUTPUT_DIR}/logs/${NAME}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
    --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    --tee 3 --log_dir ${OUTPUT_DIR}/logs/${NAME}/${WORKER_ID}"

##### Prepare logdirs #######

# NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/data_cache/"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoints/"
mkdir -p "${OUTPUT_BASEPATH}/logs/"
DEFAULT_TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}"
# 宁波集群：同任务可视化中的日志存储路径，便于开启查看Tensorboard 0411 lzd 
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${DEFAULT_TENSORBOARD_DIR}}
mkdir -p ${TENSORBOARD_DIR}

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
        --eval-interval 10000000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --tokenizer-type ${TOKENIZER_TYPE:-021Tokenizer} \
        --tokenizer-model $TOKENIZER_PATH \
        --vocab-file $TOKENIZER_PATH/tokenizer.model \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --rotary-seq-len-interpolation-factor 1 \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --moe-router-dtype fp32 \
        --moe-permute-fusion \
        --moe-router-padding-for-fp8 \
        --multi-latent-attention"
        #--no-rope-fusion \
        # --patch-tokenizer-type DeepSeekV2Tokenizer \
        # --rotary-seq-len-interpolation-factor 1 增加

# tokenizer_options=" \
#         --max-padding-length ${PAD_LEN} \
#         --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
#         "

#动态bs
ENABLE_RAMPUP_BS=${ENABLE_RAMPUP_BS:-false}
if  [[ $ENABLE_RAMPUP_BS = false ]]; then
    LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2000}
    LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    megatron_options=" ${megatron_options} \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} "
else
    warm_step=${LR_WARMUP_ITERS:-2000}
    GLOBAL_BATCH_SIZE_avg=1920
    TRAIN_SAMPLES=$(( ${TRAIN_TOKENS} / ${SEQ_LEN} ))
    LR_WARMUP_SAMPLES=$((${warm_step} * ${GLOBAL_BATCH_SIZE_avg} ))
    LR_DECAY_SAMPLES=$(( ${TRAIN_TOKENS} /  ${SEQ_LEN} ))
    megatron_options=" ${megatron_options} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --train-samples ${TRAIN_SAMPLES} \
        --rampup-batch-size 1920 960 54931640 "
fi

# Turn on PyTorchProfiler in DSW
if [ $ENV = dsw ]; then
    export CUDA_LAUNCH_BLOCKING=1
    prof_options=" --profile --use-pytorch-profiler --profile-step-end 11 --profile-ranks 0 1 2 3 4 5 6 7 "
fi

# 开启pipeline_timer，将每个rank写到对应的文件中
if [[ ${PROFILE:-off} = on ]]; then
    export PIPELINE_TIMER_LEVEL=3
    export PIPELINE_TIMER_LOG_DIR=$OUTPUT_DIR/logs/${TIMESTAMP}_${NNODES}/
    mkdir -p $PIPELINE_TIMER_LOG_DIR
fi

if [[ ${SEQWARM:-off} = on ]]; then
seqwarm_options=" --warmup-seq-length 0:2048,100:4096 "
fi

# new_options=" --checkpoint-kv-up-proj --recompute-inputlayer-rmsnorm --recompute-pre-mlp-rmsnorm "
if [[ ${CUSTOM_PIPE:-none} != none ]]; then
    new_options=" ${new_options} --custom-pipeline $CUSTOM_PIPE"
fi

# Use TP-PP-DP mapping
if [[ ${TP_PP_DP_MAP:-off} = on ]] ; then
    new_options=" ${new_options} --use-tp-pp-dp-mapping "
fi

# User custom FSDP from Megatron Core
if [[ ${USE_FSDP:-false} = true ]] ; then
    fsdp_options="\
        --use-custom-fsdp \
        --data-parallel-sharding-strategy optim_grads_params \
        --no-gradient-accumulation-fusion \
        --calculate-per-token-loss \
        "
    unset CUDA_MAX_CONNECTIONS
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi

# Precision Aware Optimizer
PAO_LEVEL=${PAO:-none}
if [[ $PAO_LEVEL = none ]]; then
    new_options=" ${new_options} \
    "
elif [[ $PAO_LEVEL = moments ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
    "
elif [[ $PAO_LEVEL = grads ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
        --main-grads-dtype bf16 \
    "
elif [[ $PAO_LEVEL = weights ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
        --main-grads-dtype bf16 \
        --main-params-dtype fp16 \
    "
else
    echo "PAO_LEVEL=${PAO_LEVEL} is not a valid option. Valid options include: none, moments, grads, weights"
    exit 1
fi

OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-false}
if [[ $OFFLOAD_OPTIMIZER = true ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --optimizer-cpu-offload \
        --overlap-cpu-optimizer-d2h-h2d \
    "
        #--use-torch-optimizer-for-cpu-offload \

fi

# 开启12LHSD的atten计算方法,打印MFU
if [[ ${BENCHMARK_MFU:-true} = true ]] ; then
    new_options=" ${new_options} \
    --num-steps-average-throughput 5 \
    --benchmark-target-tflops 1200.00 \
    --benchmark-check-begins 3000 \
    --benchmark-check-ends 5000 \
    --benchmark-pass-action continue"
fi                    

# 开启12LHSD的atten计算方法,打印MFU
if [[ ${PRINT_MFU:-true} = true ]] ; then
    new_options=" ${new_options} --use-legacy-throughput "
fi

if [[ ${CHECK_NAN:-true} = false ]]; then
    new_options=" ${new_options} --no-check-for-nan-in-loss-and-grad"
fi

if [[ ${FP8_COMM:-false} = true ]]; then
    #new_options=" ${new_options} --fp8-comm "
    export FP8_COMM_DEEPEP=true
fi

# 开启流水线并行调度plan和executer分离
if [[ ${PLAN_EXEC_SPLIT:-false} = true ]]; then
    new_options=" ${new_options} --plan-exec-split"
    SCHEDULE_TYPE=${SCHEDULE_TYPE:-1f1b}
    new_options=" ${new_options} --schedule-type ${SCHEDULE_TYPE} "
fi

# 开启调度可视化--保存可视化数据
if [[ ${SCHEDULE_VISUAL:-false} = true ]]; then
    rm -rf ${TENSORBOARD_DIR}/Timecond
    new_options=" ${new_options} --schedule-visualble-path ${TENSORBOARD_DIR}/Timecond --schedule-visual-iter-start ${SCHEDULE_VISUAL_ITER_START:-30}  --schedule-visual-iter-end ${SCHEDULE_VISUAL_ITER_END:-35}"
    if [[ ${NODE_RANK} = 0 ]]; then
        echo "Schedule visualization start !!!"
        nohup bash schedule_visual.sh ${MEGATRON_PATH} ${TENSORBOARD_DIR} ${SCHEDULE_VISUAL_ITER_END:-35} &
    fi
fi

if [[ ${ENABLE_TIMING_LOG:-false} = true ]]; then
    megatron_options=" ${megatron_options} \
        --timing-log-level 2 \
        --timing-log-option minmax \
    "
fi

if [[ ${MANUAL_GC:-0} -gt 0 ]]; then
    megatron_options=" ${megatron_options} --manual-gc --manual-gc-interval ${MANUAL_GC} "
fi

if [[ ${MTP_NUM_LAYERS:-0} -gt 0 ]]; then
    megatron_options=" ${megatron_options} --mtp-num-layers ${MTP_NUM_LAYERS} "
fi

run_cmd="torchrun $DISTRIBUTED_ARGS ${MEGATRON_PATH}/pretrain_gpt.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${fl_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} \
 ${uneven_split_option} ${prof_options} ${seqwarm_options} ${new_options} ${fsdp_options} ${ckpt_options}"

echo ${run_cmd}
[[ $RANK = 0 ]] && mkdir -p ${OUTPUT_DIR}/logs/${NAME} && echo ${run_cmd} > ${OUTPUT_DIR}/logs/${NAME}/cmd-${MODEL_SIZE}-pp-${PP}-ep-${EP}-AC-${AC}-gbs-${GLOBAL_BATCH_SIZE}-${TIMESTAMP}.sh
eval ${run_cmd}

