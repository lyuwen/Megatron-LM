#!/bin/bash
set -ex

######################################
# Change the below configurations here
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/data_cache

# DATASET=$(cat datalist_$DATAINDEX|grep -v '^#')
# DATASET_FILE="batched-training-no00/batch-data-${DATAINDEX}.txt"
# DATASET_FILE="/public/home/lfu/llm/training-data/train-datalist-test"
DATASET_FILE="${DATASET_FILE:-datalist_all}"
DATASET="$(grep -v '^#' ${DATASET_FILE})"
#OUTPUT_DIR=output

CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
if [[ ${RUN_CPT} = on ]]; then
    if [[ ! -f ${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt ]]; then
        if [[ -z "${LOAD_CHECKPOINT_PATH}" ]]; then
            echo "CPT mode requires LOAD_CHECKPOINT_PATH."
            exit 1
        fi
        CPT_CONFIG="\
               --reset-dataloader \
               --reset-iterations \
               --no-load-optim \
               --no-load-rng \
               "
    fi
fi
if [[ -z "${LOAD_CHECKPOINT_PATH}" ]]; then
  LOAD_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
fi

if [[ ${TOKENIZER_TYPE:-021Tokenizer} == 021Tokenizer ]]; then
  if [[ -z "${TOKENIZER_PATH}" ]]; then
    echo "Missing TOKENIZER_PATH"
    exit 1
  fi
  TOKENIZER_CONFIG="\
      --tokenizer-type 021Tokenizer \
      --tokenizer-model $TOKENIZER_PATH \
      --vocab-file $TOKENIZER_PATH/tokenizer.model \
      "
elif [[ ${TOKENIZER_TYPE} == Huggingface ]]; then
  TOKENIZER_CONFIG="\
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model $TOKENIZER_PATH \
      "
elif [[ -z ${TOKENIZER_CONFIG} ]]; then
  echo "Unknown tokenizer type, set TOKENIZER_CONFIG on your own."
  exit 1
fi


if [[ ! -z ${TIMESTAMP} ]]; then
  DATETIME=${TIMESTAMP}
else
  DATETIME="$(date +'%Y%m%dT%H%M')"
fi
TENSORBOARD_PATH="$OUTPUT_DIR/tf_logs"

TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}

GPUS_PER_NODE=${GPUS_PER_NODE:-${TQ_GPU_NUM}}
NNODES=$WORLD_SIZE

NEW_PARAMS=""
if [[ ! -z ${USE_NEW_PARAMS} ]];
then
    NEW_PARAMS="\
        --use-tp-pp-dp-mapping \
        "
fi

if [[ ! -z ${MMDATASET} ]];
then
    NEW_PARAMS=" ${NEW_PARAMS} \
        --use-gpt-dataset-mm \
        "
fi

MODEL_SIZE=${MODEL_SIZE:-2B}
if [[ ${MODEL_SIZE} = "2B" ]]; then
  HIDDEN_SIZE=3072 
  FFN_HIDDEN_SIZE=8192 
  NUM_LAYERS=16
  NUM_HEADS=24
  GQA_CONFIG=" "
elif [[ ${MODEL_SIZE} = "1B" ]]; then
  HIDDEN_SIZE=2048 
  FFN_HIDDEN_SIZE=16384
  NUM_LAYERS=16
  NUM_HEADS=16
  GQA_CONFIG=" "
elif [[ ${MODEL_SIZE} = "32B" ]]; then
  HIDDEN_SIZE=5120
  FFN_HIDDEN_SIZE=27648
  NUM_LAYERS=64
  NUM_HEADS=40
  NUM_KV_HEADS=8
  GQA_CONFIG=" \
      --group-query-attention \
      --num-query-groups $NUM_KV_HEADS \
      "
  MAX_POSITION_EMBEDDINGS=32768
fi


SEQ_LENGTH=${SEQ_LENGTH:-2048}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2048}  # e.g. llama: 4M tokens

LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-2000}
# LR_WARMUP_FRAC=0.0006
WEIGHT_DECAY=0.1
GRAD_CLIP=1
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}

LOG_LEVEL=${LOG_LEVEL:-20} # 20 - INFO


AC=${AC:-none} # full
if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi
if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    ACTIVATION_CHECKPOINT_OPTIONS=" \
        --recompute-method uniform \
        --recompute-num-layers ${MP_AC_LAYERS} \
        --recompute-granularity full"
elif [ $AC = sel ]; then
    ACTIVATION_CHECKPOINT_OPTIONS=" \
        --recompute-activations"
elif [ $AC = none ]; then
    ACTIVATION_CHECKPOINT_OPTIONS=" "
fi

# User custom FSDP from Megatron Core
if [[ ${USE_FSDP:-false} = true ]] ; then
    FSDP_OPTIONS="\
        --use-custom-fsdp \
        --data-parallel-sharding-strategy optim_grads_params \
        --no-gradient-accumulation-fusion \
        --calculate-per-token-loss \
        "
    unset CUDA_MAX_CONNECTIONS
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi


# EXTRA_VALID="/public/home/lfu/llm/training-data/validation/valid-datalist"
if [[ ! -z ${EXTRA_VALID} ]]; then
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
fi
if [[ -z ${TOTAL_STEPS} ]]; then
  TOTAL_STEPS=2400000
fi

TRAIN_STEPS=${TRAIN_STEPS:-100}

SAMPLE_SIZE="$(($(python3 sum_row1.py ${DATASET_FILE})*990/1000/${SEQ_LENGTH}))"
SAMPLE_ITERS="$((${SAMPLE_SIZE}/${GLOBAL_BATCH_SIZE}))"
SEEN_STEPS=0
TRAIN_STEPS=$((${SEEN_STEPS} + ${SAMPLE_ITERS}))
TOTAL_STEPS=$TRAIN_STEPS

echo "$( date +"%Y-%m-%dT%H:%M:%S(%Z)") - CRITICAL - Using dataset file: ${DATASET_FILE}, with ${SAMPLE_SIZE} samples" | tee -a ${OUTPUT_DIR}/data.log
echo "$( date +"%Y-%m-%dT%H:%M:%S(%Z)") - CRITICAL - Train ${SAMPLE_ITERS} steps from step $SEEN_STEPS to ${TRAIN_STEPS} with total $TOTAL_STEPS steps planned." | tee -a ${OUTPUT_DIR}/data.log

if [[ ! -z "$DL_WORKERS" ]]; then
    DL_OPTIONS=" --num-workers ${DL_WORKERS} "
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
  --rdzv_id=333 --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --tee 3 --log_dir ${OUTPUT_DIR}/logs/${DATETIME}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank ${RANK} \
  --tee 3 --log_dir ${OUTPUT_DIR}/logs/${DATETIME}"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --context-parallel-size ${CP} \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       ${GQA_CONFIG} \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS:-${SEQ_LENGTH}} \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $LOAD_CHECKPOINT_PATH \
       --data-path $DATASET \
       --data-suffix ".npy" \
       ${TOKENIZER_CONFIG} \
       --split 989,10,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --lr-decay-iters $TOTAL_STEPS \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --lr-warmup-init $MIN_LR \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --save-interval ${SAVE_INTERVAL} \
       --log-interval 1 \
       --log-throughput \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --timing-log-level 1 \
       --eval-interval ${EVAL_INTERVAL} \
       --eval-iters 10 \
       --bf16 \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization RMSNorm \
       --disable-bias-linear \
       --add-qkv-bias \
       --tensorboard-dir $TENSORBOARD_PATH \
       --tensorboard-log-interval 1 \
       --use-flash-attn \
       --timing-log-level 1 \
       --use-mcore-models \
       --no-masked-softmax-fusion \
       --attention-softmax-in-fp32 \
       --distributed-timeout-minutes 180 \
       --num-dataset-builder-threads 2 \
       --data-cache-path $OUTPUT_DIR/data_cache \
       --logging-level ${LOG_LEVEL} \
       --use-distributed-optimizer \
       --overlap-param-gather \
       --overlap-grad-reduce \
       --async-save \
       ${CPT_CONFIG} \
       ${NEW_PARAMS} \
       ${DL_OPTIONS} \
       ${EXTRA_VALID_ARGS} \
       ${ACTIVATION_CHECKPOINT_OPTIONS} \
       ${USE_FSDP} \
       --force-train-samples ${SAMPLE_SIZE}
       # --renormalize-blend-weights \
