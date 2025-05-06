#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
FIRST_LAYERS_PP=$6
LAST_LAYERS_PP=$7
MP_VP=$8 #
EP=$9
PR=${10}
USE_TE=true # USE_TE=$8 
MG2HF=${11}
HF_CKPT_PATH=${12}
ITERATION=${13}
CHECK_DIFF=${14}
LOW_MEMORY=${15}
SAVE_NUM_FILES=${16}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# MEGATRON_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
MEGATRON_PATH="/mnt/cpfs/users/mzy/moe/ZJ-Megatron-LM-0.0.11b1-20250328_3"
# export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}
echo $PYTHONPATH

FL=true
if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    fl_options=" --attention-backend flash "
elif [ $FL = false ]; then
    # export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    fl_options=" --attention-backend unfused "
fi

if [ $MODEL_SIZE = A2.4B ]; then

HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=27
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=2400
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

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    "


cpu_options=" \
            --use-cpu-initialization"


elif [ $MODEL_SIZE = DSnew16B_v2_20250409 ]; then
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=28
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
MOE_FIRST_K_DENSE_REPLACE=0

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --target-expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 1.0 \
    --attention-backend flash \
    "
    # 
    # --use-flash-attn \
    # --first_k_dense_replace 0 \
    # --moe-shared-expert-overlap \

cpu_options=" \
            --use-cpu-initialization"


elif [ $MODEL_SIZE = A21B ]; then

HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=128
NUM_HIDDEN_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=163840
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
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    "

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = config1-A0.6B ]; then

HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=16
INTERMEDIATE_SIZE=6144
MOE_INTERMEDIATE_SIZE=768
MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=256 # 256
Q_LORA_RANK=256
KV_LORA_RANK=128
QK_NOPE_HEAD_DIM=64
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=64
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    "
    # --use-flash-attn \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DS16B ]; then

HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=28
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=256 # 256
# Q_LORA_RANK=256
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
MOE_AUX_LOSS_COEFF=1e-3

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --first_k_dense_replace 0 \
    "
    # --use-flash-attn \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DS2.5B ]; then

HIDDEN_SIZE=1280
NUM_ATTENTION_HEADS=10
NUM_HIDDEN_LAYERS=9
INTERMEDIATE_SIZE=8192
MOE_INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=256 # 256
# Q_LORA_RANK=256
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
RMS_NORM_EPS=1e-6
MOE_AUX_LOSS_COEFF=1e-3

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --first_k_dense_replace 1 \
    "
    # --use-flash-attn \
    # --first_k_dense_replace 0 \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DS200B ]; then

HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=128
NUM_HIDDEN_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=256 # 256
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40 # routed scaling factor/ rope scaling factor, diffs, ref https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json
NUM_EXPERTS=160
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6
MOE_AUX_LOSS_COEFF=1e-3

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --first_k_dense_replace 0 \
    --q-lora-rank ${Q_LORA_RANK} \
    "
    # --use-flash-attn \
    # --first_k_dense_replace 0 \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DSnew2.5B ]; then

HIDDEN_SIZE=1280
NUM_ATTENTION_HEADS=10
NUM_HIDDEN_LAYERS=9
INTERMEDIATE_SIZE=8192
MOE_INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=4096
EXTRA_VOCAB_SIZE=256 # 256
# Q_LORA_RANK=256
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
RMS_NORM_EPS=1e-6
MOE_AUX_LOSS_COEFF=1e-3
MOE_FIRST_K_DENSE_REPLACE=2

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --target-expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 0.001 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-grouped-gemm \
    --attention-backend flash \
    "
    # 
    # --use-flash-attn \
    # --first_k_dense_replace 0 \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DSnew2.5B-baseline ]; then

HIDDEN_SIZE=1280
NUM_ATTENTION_HEADS=10
NUM_HIDDEN_LAYERS=12
INTERMEDIATE_SIZE=8192
MOE_INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=4096
EXTRA_VOCAB_SIZE=256 # 256
# Q_LORA_RANK=256
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
RMS_NORM_EPS=1e-6
MOE_AUX_LOSS_COEFF=1e-3
MOE_FIRST_K_DENSE_REPLACE=1 ## 

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --target-expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-router-score-function softmax \
    --moe-router-pre-softmax \
    --moe-router-load-balancing-type aux_loss \
    --attention-backend flash \
    --moe-grouped-gemm \
    "
    # 
    # --use-flash-attn \
    # --first_k_dense_replace 0 \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DSnew16B ]; then

HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=28
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=4096
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
MOE_FIRST_K_DENSE_REPLACE=1

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --target-expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 0.001 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-grouped-gemm \
    --attention-backend flash \
    "
    # 
    # --use-flash-attn \
    # --first_k_dense_replace 0 \
    # --moe-shared-expert-overlap \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DSnew16B_v2 ]; then

HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=28
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
MOE_FIRST_K_DENSE_REPLACE=0

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --target-expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type allgather \
    --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 1.0 \
    --attention-backend flash \
    "
    # 
    # --use-flash-attn \
    # --first_k_dense_replace 0 \
    # --moe-shared-expert-overlap \

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = DSnew200B ]; then

HIDDEN_SIZE=5120
# HIDDEN_SIZE=256
NUM_ATTENTION_HEADS=128
NUM_HIDDEN_LAYERS=60
# NUM_LAYERS=60
# NUM_LAYERS=64
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=4096
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
    --target-expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-grouped-gemm \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-bias-update-rate 1e-3 \
    --moe-token-drop-policy probs \
    --attention-backend flash"

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = ZJ_DS_aux200B ]; then

HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=128
NUM_HIDDEN_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=163840
EXTRA_VOCAB_SIZE=256
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
RMS_NORM_EPS=1e-6
MOE_FIRST_K_DENSE_REPLACE=1
MOE_AUX_LOSS_COEFF=1e-3
ROUTED_SCALING_FACTOR=1


moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --target-expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-grouped-gemm \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-pre-softmax \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-score-function softmax
    --attention-backend flash"

cpu_options=" \
            --use-cpu-initialization"

fi


if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

elif [ $MG2HF = false ]; then
    convert_options=""
fi

if [ $USE_TE = true ]; then
    te_options=" \
                --transformer-impl transformer_engine \
                "

elif [ $USE_TE = false ]; then
    te_options=" \
                --transformer-impl local \
                "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"

fi

# target vp
if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --target-num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi


if [ ${LOW_MEMORY} = true ]; then
    convert_options="${convert_options} \
                     --use-low-memory-convert \
                     --save-num-files ${SAVE_NUM_FILES} "
fi


if [ ${CHECK_DIFF} = true ]; then
    check_options="--check-diff"
else
    check_options=""
fi



DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
TOKENIZER_PATH=/mnt/cpfs/code/tokenizers/zjllm-llama3-tokenizer
#TOKENIZER_PATH=/mnt/data/lufanfeng/Megatron-Core/zjllm-llama3-tokenizer

torchrun ${DISTRIBUTED_ARGS} hf2mcore_deepseek_v2_moe_v3.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --target-decoder-first-pipeline-num-layers ${FIRST_LAYERS_PP} \
    --target-decoder-last-pipeline-num-layers ${LAST_LAYERS_PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --swiglu \
    --num-layers ${NUM_HIDDEN_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --untie-embeddings-and-output-weights \
    --no-bias-swiglu-fusion \
    --position-embedding-type rope \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --rotary-base ${ROPE_THETA} \
    --rotary-scaling-factor ${SCALE_FACTOR} \
    --qk-layernorm \
    --kv-channels ${V_HEAD_DIM} \
    --multi-latent-attention \
    ${moe_options} \
    ${te_options} \
    ${convert_options} \
    ${pr_options} \
    ${cpu_options} \
    ${vp_options} \
    ${fl_options} \
    --tokenizer-type 021Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --vocab-file $TOKENIZER_PATH/tokenizer.model \
    --iteration ${ITERATION} \
    ${check_options}
    # --patch-tokenizer-type LLama3Tokenizer \
    # --extra-vocab-size ${EXTRA_VOCAB_SIZE} 

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
