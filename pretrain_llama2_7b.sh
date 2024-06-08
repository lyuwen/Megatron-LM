#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex

#apt update && apt install -y libaio-dev

# pip install sentencepiece transformers deepspeed tiktoken blobfile -i https://pypi.tuna.tsinghua.edu.cn/simple

######################################
# Change the below configurations here
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p ${OUTPUT_DIR}

# DATASET=$(cat datalist_$DATAINDEX|grep -v '^#')
DATASET_FILE="batched-training-no00/batch-data-${DATAINDEX}.txt"
DATASET="$(grep -v '^#' ${DATASET_FILE})"
#OUTPUT_DIR=output

CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
# LOAD_CHECKPOINT_PATH=${CHECKPOINT_PATH}
if [[ -z "${LOAD_CHECKPOINT_PATH}" ]]; then
  LOAD_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
fi
# TOKENIZER_PATH=./zjllm-llama3-tokenizer # offical llama tokenizer.model
TOKENIZER_PATH=/mnt/nfs66/home/lfu/zjllm-lfu/tokenizer/zjllm-llama3-tokenizer

#DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
DATETIME="$(date +'%Y%m%dT%H%M%S')"
TENSORBOARD_PATH="$OUTPUT_DIR/tf_logs"

# DATALOADER_LOG=$OUTPUT_DIR/dataloader.log
# RESET_DATALOADER="--reset-dataloader --override-opt_param-scheduler"
# if [[ -f ${DATALOADER_LOG} ]]; then
#   # grep "${DATASET_FILE}" ${DATALOADER_LOG}) > /dev/null && RESET_DATALOADER=""
#   if [[ "$(tail -n 1 ${DATALOADER_LOG})" = ${DATASET_FILE} ]]; then
#     RESET_DATALOADER=""
#   fi
# else
#   if [[ ! -z "${NO_RESET_FIRST}" ]]; then
#     RESET_DATALOADER=""
#   fi
# fi
# echo ${DATASET_FILE} >> ${DATALOADER_LOG}

if [[ -z ${LOG_DIR} ]];
then
  LOG_DIR="$OUTPUT_DIR"
fi

TP=2
PP=2

GPUS_PER_NODE=${TQ_GPU_NUM}
# MASTER_ADDR=$MASTER_ADDR
# MASTER_PORT=$MASTER_PORT
NNODES=$WORLD_SIZE
# NODE_RANK=0

HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
NUM_LAYERS=32 # e.g. llama-13b: 40
NUM_HEADS=32 # e.g. llama-13b: 40
SEQ_LENGTH=2048
#NUM_KV_HEADS=32 # llama2 70B uses GQA

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=2048  # e.g. llama: 4M tokens
# TRAIN_STEPS=60000 #250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1
SAVE_INTERVAL=10000

# if [[ -z ${SEEN_STEPS} ]]; then
#   SEEN_STEPS=$([[ -f ${LOAD_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt ]] && cat ${LOAD_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt || echo 0)
# fi
# SAMPLE_SIZE="$(($(python3 sum_row1.py ${DATASET_FILE})*940/1000/${SEQ_LENGTH}))"
# SAMPLE_ITERS="$((${SAMPLE_SIZE}/${GLOBAL_BATCH_SIZE}))"
# TRAIN_STEPS=$((${SEEN_STEPS} + ${SAMPLE_ITERS}))
#
if [[ -z ${RESET_ITERATIONS} ]]; then
  eval "$(python check_restart_status.py \
    --lbc ${LOAD_CHECKPOINT_PATH} \
    --cbc ${CHECKPOINT_PATH} \
    --dl ${DATASET_FILE} \
    --save ${OUTPUT_DIR}/restart-status.yaml \
    --seq-len ${SEQ_LENGTH} \
    --tsr 0.949 \
    --gbs ${GLOBAL_BATCH_SIZE} \
    )"
else
  eval "$(python check_restart_status.py \
    --lbc ${LOAD_CHECKPOINT_PATH} \
    --cbc ${CHECKPOINT_PATH} \
    --dl ${DATASET_FILE} \
    --save ${OUTPUT_DIR}/restart-status.yaml \
    --seq-len ${SEQ_LENGTH} \
    --tsr 0.949 \
    --gbs ${GLOBAL_BATCH_SIZE} \
    --ri \
    )"
fi
#
if [[ -z ${TOTAL_STEPS} ]]; then
  TOTAL_STEPS=3600000
fi

echo "$( date +"%Y-%m-%dT%H:%M:%S(%Z)") - CRITICAL - Using dataset file: ${DATASET_FILE}, with ${SAMPLE_SIZE} samples" | tee -a ${OUTPUT_DIR}/data.log
echo "$( date +"%Y-%m-%dT%H:%M:%S(%Z)") - CRITICAL - Train ${SAMPLE_ITERS} steps from step $SEEN_STEPS to ${TRAIN_STEPS} with total $TOTAL_STEPS steps planned." | tee -a ${OUTPUT_DIR}/data.log

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="false"
# activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################


# # debug
#GPUS_PER_NODE=8
#NNODES=1
#MASTER_ADDR=localhost
#MASTER_PORT=12345

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
  --rdzv_id=333 --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --tee 3 --log_dir ${LOG_DIR}/logs/${DATETIME}"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $LOAD_CHECKPOINT_PATH \
       --data-path $DATASET \
       --data-suffix ".npy" \
       --tokenizer-type TikTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --vocab-file $TOKENIZER_PATH/tokenizer.model \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --lr-decay-iters $TOTAL_STEPS \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --save-interval ${SAVE_INTERVAL} \
       --log-interval 1 \
       --log-throughput \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --timing-log-level 1 \
       --eval-interval 300 \
       --eval-iters 10 \
       --bf16 \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --no-position-embedding \
       --swiglu \
       --normalization RMSNorm \
       --disable-bias-linear \
       --tensorboard-dir $TENSORBOARD_PATH \
       --tensorboard-log-interval 1 \
       --use-flash-attn \
       --timing-log-level 1 \
       --use-mcore-models \
       --no-masked-softmax-fusion \
       --attention-softmax-in-fp32 \
       --distributed-timeout-minutes 180 \
       --num-dataset-builder-threads 2 \
       --data-cache-path $OUTPUT_DIR/data_cache_$DATAINDEX \
       ${RESET_DATALOADER} \
       --force-train-samples ${SAMPLE_SIZE}
       #--load $CHECKPOINT_PATH \
       #--no-load-optim \
       #--no-load-rng \
      #  --num-key-value-heads $NUM_KV_HEADS \
      #  --no-pipeline-parallel \
      #  --no-query-key-layer-scaling \
       #--override-lr-scheduler \
       #--no-bias-dropout-fusion \
