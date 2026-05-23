#!/bin/bash
# Training launch script for Layered Video Generation
# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
#
# Usage:
#   ./run_train.sh --model_path /path/to/Wan2.1 --data_root /path/to/data
#
# Options:
#   --config            Config file path (default: training/configs/default.yaml)
#   --num_gpus          Number of GPUs (default: 4)
#   --model_path        Path to Wan2.1 checkpoint
#   --data_root         Path to training data
#   --layer_offset_mode Layer position offset mode: none, fixed, learnable (default: learnable)
#   --mask_mode         Mask processing mode: vae, downsample, downsample-project, vae-project, vae-lora, mask-vae-project, mask-vae-joint (default: vae)
#   --mask_vae_lora_path Path to mask VAE LoRA checkpoint (vae-lora mode)
#   --output_dir        Output directory (default: ./outputs)
#   --resume_from       Resume from checkpoint path

set -e

# Default values
CONFIG_FILE="${CONFIG_FILE:-training/configs/default.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${MODEL_PATH:-/path/to/Wan2.1}"
DATA_ROOT="${DATA_ROOT:-/path/to/dataset}"
LAYER_OFFSET_MODE="${LAYER_OFFSET_MODE:-learnable}"
MASK_MODE="${MASK_MODE:-vae}"
MASK_VAE_LORA_PATH="${MASK_VAE_LORA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
RESUME_FROM="${RESUME_FROM:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --layer_offset_mode)
            LAYER_OFFSET_MODE="$2"
            shift 2
            ;;
        --mask_mode)
            MASK_MODE="$2"
            shift 2
            ;;
        --mask_vae_lora_path)
            MASK_VAE_LORA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Layered Video Generation Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model path: $MODEL_PATH"
echo "Data root: $DATA_ROOT"
echo "Layer offset mode: $LAYER_OFFSET_MODE"
echo "Mask mode: $MASK_MODE"
if [ -n "$MASK_VAE_LORA_PATH" ]; then
    echo "Mask VAE LoRA path: $MASK_VAE_LORA_PATH"
fi
echo "Output dir: $OUTPUT_DIR"
if [ -n "$RESUME_FROM" ]; then
    echo "Resume from: $RESUME_FROM"
fi
echo "=========================================="

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate is not installed. Install with: pip install accelerate"
    exit 1
fi

# Build common arguments
TRAIN_ARGS="--config $CONFIG_FILE \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --layer_offset_mode $LAYER_OFFSET_MODE \
    --mask_mode $MASK_MODE \
    --output_dir $OUTPUT_DIR"

# Add resume_from if provided
if [ -n "$RESUME_FROM" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume_from $RESUME_FROM"
fi
if [ -n "$MASK_VAE_LORA_PATH" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --mask_vae_lora_path $MASK_VAE_LORA_PATH"
fi

# Multi-GPU training with Accelerate
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    accelerate launch \
        --multi_gpu \
        --num_processes "$NUM_GPUS" \
        --mixed_precision bf16 \
        training/train_layered.py \
        $TRAIN_ARGS
else
    echo "Starting single-GPU training..."
    python training/train_layered.py \
        $TRAIN_ARGS
fi

echo "Training completed!"
