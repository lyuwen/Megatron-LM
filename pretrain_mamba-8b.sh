#!/bin/bash

# Use: ./train.sh <data-path> <tokenizer-path>

mkdir -p ${OUTPUT_DIR}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}

GPUS_PER_NODE=${TQ_GPU_NUM}
NNODES=$WORLD_SIZE
MODEL_SCALE=${MODEL_SCALE:-"8B"} # or "8B"
set -x

case "${MODEL_SCALE}" in
    "800M")
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=1024
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
        ;;
    "1600M")
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=2048
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
        ;;
    "2800M")
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=2560
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
        ;;
    "8B")
        TENSOR_MODEL_PARALLEL_SIZE=4
        NUM_LAYERS=56
        HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
        ;;
    "custom")
        echo TENSOR_MODEL_PARALLEL_SIZE : $TENSOR_MODEL_PARALLEL_SIZE
        echo NUM_LAYERS                 : $NUM_LAYERS
        echo HIDDEN_SIZE                : $HIDDEN_SIZE
        echo NUM_ATTENTION_HEADS        : $NUM_ATTENTION_HEADS
        echo GLOBAL_BATCH_SIZE          : $GLOBAL_BATCH_SIZE
        ;;
    *)
        echo "Invalid version specified"
        exit 1
        ;;
esac

DATASET_FILE="/public/home/lfu/llm/training-data/train-datalist-test"
DATA_PATH="$(grep -v '^#' ${DATASET_FILE})"
TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/nfs66/home/lfu/zjllm-lfu/tokenizer/zjllm-llama3-tokenizer}
TOKENIZER_PATH=/public/home/lfu/llm/framework/tokenizer/zjllm-llama3-tokenizer

DATETIME="$(date +'%Y%m%dT%H%M%S')"

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
DATACACHE_DIR="${OUTPUT_DIR}/data-cache"
# TENSORBOARD_DIR="./tensorboard"
TENSORBOARD_DIR="${OUTPUT_DIR}/tf_logs"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="${OUTPUT_DIR}/triton-cache/"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

SPLIT="949,50,1"
MICRO_BATCH_SIZE=4
LR="2.5e-4"
MINLR="2.5e-5"
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

SEQ_LENGTH=4096
# TRAIN_SAMPLES=73242188  # 300B tokens / 4096
# LR_WARMUP_SAMPLES=50000
# LR_DECAY_SAMPLES=73192188 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES
SAVE_INTERVAL=1000
EVAL_INTERVAL=10
EVAL_ITERS=10

LOG_LEVEL=${LOG_LEVEL:-20} # 20 - INFO

SAMPLE_SIZE="$(($(python3 sum_row1.py ${DATASET_FILE})*940/1000/${SEQ_LENGTH}))"
SAMPLE_ITERS="$((${SAMPLE_SIZE}/${GLOBAL_BATCH_SIZE}))"
# TRAIN_STEPS=$((${SEEN_STEPS} + ${SAMPLE_ITERS}))
TRAIN_STEPS=${SAMPLE_ITERS}
TRAIN_SAMPLES=${SAMPLE_SIZE}
LR_WARMUP_STEPS=2000
LR_WARMUP_SAMPLES=$((${LR_WARMUP_STEPS}*${GLOBAL_BATCH_SIZE}))
LR_DECAY_SAMPLES=$((${TOTAL_STEPS}*${GLOBAL_BATCH_SIZE}))


EXTRA_VALID="/public/home/lfu/llm/training-data/validation/valid-datalist"
EXTRA_VALID_ARGS=" \
    --extra-valid-datalist ${EXTRA_VALID} \
    --extra-valid-data-samples $(($(python sum_row1.py ${EXTRA_VALID})/${SEQ_LENGTH})) \
    --extra-valid-data-names   "dolma-test" \
    --extra-valid-datalist ${EXTRA_VALID}-c4_en --extra-valid-data-samples $((993523/${SEQ_LENGTH})) --extra-valid-data-names c4_en \
    --extra-valid-datalist ${EXTRA_VALID}-dolma_books --extra-valid-data-samples $((486667/${SEQ_LENGTH})) --extra-valid-data-names dolma_books \
    --extra-valid-datalist ${EXTRA_VALID}-dolma_cc --extra-valid-data-samples $((479817/${SEQ_LENGTH})) --extra-valid-data-names dolma_cc \
    --extra-valid-datalist ${EXTRA_VALID}-dolma_pes2o --extra-valid-data-samples $((512145/${SEQ_LENGTH})) --extra-valid-data-names dolma_pes2o \
    --extra-valid-datalist ${EXTRA_VALID}-dolma_reddit --extra-valid-data-samples $((481997/${SEQ_LENGTH})) --extra-valid-data-names dolma_reddit \
    --extra-valid-datalist ${EXTRA_VALID}-dolma_stack --extra-valid-data-samples $((416115/${SEQ_LENGTH})) --extra-valid-data-names dolma_stack \
    --extra-valid-datalist ${EXTRA_VALID}-dolma_wiki --extra-valid-data-samples $((495623/${SEQ_LENGTH})) --extra-valid-data-names dolma_wiki \
    --extra-valid-datalist ${EXTRA_VALID}-ice --extra-valid-data-samples $((896641/${SEQ_LENGTH})) --extra-valid-data-names ice \
    --extra-valid-datalist ${EXTRA_VALID}-m2d2_s2orc --extra-valid-data-samples $((979972/${SEQ_LENGTH})) --extra-valid-data-names m2d2_s2orc \
    --extra-valid-datalist ${EXTRA_VALID}-pile --extra-valid-data-samples $((666929/${SEQ_LENGTH})) --extra-valid-data-names pile \
    --extra-valid-datalist ${EXTRA_VALID}-wikitext_103 --extra-valid-data-samples $((247189/${SEQ_LENGTH})) --extra-valid-data-names wikitext_103 \
"

options=" \
       --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --sequence-parallel \
       --pipeline-model-parallel-size 1 \
       --use-distributed-optimizer \
       --overlap-param-gather \
       --overlap-grad-reduce \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type none \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATTENTION_HEADS} \
       --group-query-attention \
       --num-query-groups 8 \
       --hybrid-attention-ratio 0.08 \
       --hybrid-mlp-ratio 0.5 \
       --seq-length ${SEQ_LENGTH} \
       --max-position-embeddings ${SEQ_LENGTH} \
       --train-samples ${TRAIN_SAMPLES} \
       --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
       --lr-decay-samples ${LR_DECAY_SAMPLES} \
       --save ${CHECKPOINT_DIR} \
       --load ${CHECKPOINT_DIR} \
       --data-path ${DATA_PATH} \
       --data-cache-path ${DATACACHE_DIR} \
       --split ${SPLIT} \
       --tokenizer-type TikTokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --vocab-file $TOKENIZER_PATH/tokenizer.model \
       --distributed-backend nccl \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --lr ${LR} \
       --min-lr ${MINLR} \
       --lr-decay-style cosine \
       --weight-decay ${WEIGHT_DECAY} \
       --clip-grad $GRAD_CLIP \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --log-throughput \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --timing-log-level 1 \
       --save-interval ${SAVE_INTERVAL} \
       --eval-interval ${EVAL_INTERVAL} \
       --eval-iters ${EVAL_ITERS:-10} \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
       --no-create-attention-mask-in-dataloader \
       ${EXTRA_VALID_ARGS} \
       --logging-level ${LOG_LEVEL} \
       --tensorboard-log-interval 1 \
       --tensorboard-dir ${TENSORBOARD_DIR}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
  --rdzv_id=333 --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --tee 3 --log_dir ${OUTPUT_DIR}/logs/${DATETIME}"

torchrun ${DISTRIBUTED_ARGS} ./pretrain_mamba.py ${options}
