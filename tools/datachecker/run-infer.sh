accelerate launch --config-file accelerate_config_fsdp.yml \
  run-valid.py \
    --model-path /mnt/cpfs/training/pretrain/output/lfu-14b-pretrain-v8-1/checkpoints-converted/hf/iter_0044000/ \
    --megatron-ckpt-path /mnt/cpfs/training/pretrain/output/lfu-14b-pretrain-v8-1/checkpoints \
    --batch-size 1440 \
    --infer_batch 4 \
    --distributed True \
    -i /mnt/workspace/training/pretrain/output/lfu-14b-pretrain-v8-1/outlier-samples.txt
