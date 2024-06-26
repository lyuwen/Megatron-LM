#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=test/output
# VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=<Specify path to file>/gpt2-merges.txt
# DATA_PATH=<Specify path and file prefix>_text_document
DATA_PATH="$(cat data-files.txt)"
# A list of files containing file paths without suffix

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 12 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --lr 0.00005 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-mcore-models \
    --use-flash-attn
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
    --tokenizer-type TikTokenizer \
    --tokenizer-model /mnt/geogpt-gpfs/llm-course/home/lfu/llm/framework/tokenizer/zjllm-llama3-tokenizer \
    --vocab-file /mnt/geogpt-gpfs/llm-course/home/lfu/llm/framework/tokenizer/zjllm-llama3-tokenizer/tokenizer.model \
    --data-suffix ".npy"
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --tensorboard-dir ${CHECKPOINT_PATH}/tf_logs \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-world-size-to-tensorboard \
    --log-throughput \
    --timing-log-level 2
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
