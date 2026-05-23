#!/bin/bash

# 设置使用的 GPU 数量 (例如 8 卡)
NUM_GPUS=8

# 设置路径变量
JSONL_PATH="/home/notebook/data/group/ckr/Data/layer/layert2v_backup0109/test_data/anno_1obj_200sample.jsonl"
MODEL_PATH="/home/notebook/data/group/ckr/ckpt/Wan2.1-T2V-1.3B"
LORA_PATH="/home/notebook/data/group/ckr/Data/layer/layert2v_0116/outputs/layered-video-16ksample/checkpoints/step-6000"
MASK_VAE_LORA_PATH="/home/notebook/data/group/ckr/Data/layer/layert2v_0116/outputs/mask_vae_lora_pretrained/mask_vae_lora.pt"
OUTPUT_DIR="./test_outputs/16k-6kstep-infer"

# 运行 torchrun
# --nproc_per_node: 使用的 GPU 数量
# --master_port: 防止端口冲突，随机指定一个
torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 scripts/batch_inference.py \
    --jsonl_path "$JSONL_PATH" \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --mask_vae_lora_path "$MASK_VAE_LORA_PATH" \
    --mask_mode vae-lora \
    --no_4d_rope \
    --output_dir "$OUTPUT_DIR" \
    --width 336 \
    --height 192 \
    --frames 9 \
    --steps 50 \
    --seed 42 \
    --fps 8