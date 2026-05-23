#!/bin/bash
# Inference script for Layered Video Generation
# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
#
# Usage:
#   ./run_inference.sh --model_path /path/to/Wan2.1 --lora_path /path/to/checkpoint --prompt "Your prompt"
#
# Options:
#   --model_path        Path to Wan2.1 checkpoint
#   --lora_path         Path to LoRA checkpoint directory
#   --ema               Use EMA weights (better quality)
#   --layer_offset_mode Layer position offset mode: none, fixed, learnable (default: learnable)
#   --mask_mode         Mask processing mode: vae, downsample, downsample-project, vae-project, vae-lora, mask-vae-project, mask-vae-joint (default: vae)
#   --mask_vae_path     Path to MaskVAE checkpoint (mask-vae-project/mask-vae-joint)
#   --mask_vae_proj_path Path to projection layers (vae-project/mask-vae-project/mask-vae-joint)
#   --mask_vae_lora_path Path to VAE LoRA checkpoint (vae-lora)
#   --use_4d_rope       Use 4D RoPE (L, T, H, W) position encoding (default: true)
#   --no_4d_rope        Use original 3D RoPE (T, H, W) position encoding
#   --rope_dim_ratios   4D RoPE dimension allocation (L,T,H,W), e.g. '8,42,40,38'
#   --output_dir        Output directory (default: ./outputs)
#   --prompt            Main scene prompt
#   --fg_prompt         Foreground object prompt
#   --bg_prompt         Background scene prompt
#   --width             Video width (default: 672)
#   --height            Video height (default: 384)
#   --frames            Number of frames (default: 9)
#   --steps             Diffusion steps (default: 50)
#   --seed              Random seed (-1 for random)
#   --fps               Output FPS (default: 8)

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-/inspire/ssd/project/medical-image/250041225002/models/wan-2.1-1.3B}"
LORA_PATH="${LORA_PATH:-}"
USE_EMA=false
LAYER_OFFSET_MODE="${LAYER_OFFSET_MODE:-learnable}"
MASK_MODE="${MASK_MODE:-vae}"
MASK_VAE_PATH="${MASK_VAE_PATH:-}"
MASK_VAE_PROJ_PATH="${MASK_VAE_PROJ_PATH:-}"
MASK_VAE_LORA_PATH="${MASK_VAE_LORA_PATH:-}"
USE_4D_ROPE=true
ROPE_DIM_RATIOS="${ROPE_DIM_RATIOS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
PROMPT="${PROMPT:-A ship sails on the ocean under blue sky.}"
FG_PROMPT="${FG_PROMPT:-A ship.}"
BG_PROMPT="${BG_PROMPT:-A vast ocean under blue sky.}"
WIDTH="${WIDTH:-672}"
HEIGHT="${HEIGHT:-384}"
FRAMES="${FRAMES:-9}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-0}"
FPS="${FPS:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --ema)
            USE_EMA=true
            shift
            ;;
        --layer_offset_mode)
            LAYER_OFFSET_MODE="$2"
            shift 2
            ;;
        --mask_mode)
            MASK_MODE="$2"
            shift 2
            ;;
        --mask_vae_path)
            MASK_VAE_PATH="$2"
            shift 2
            ;;
        --mask_vae_proj_path)
            MASK_VAE_PROJ_PATH="$2"
            shift 2
            ;;
        --mask_vae_lora_path)
            MASK_VAE_LORA_PATH="$2"
            shift 2
            ;;
        --use_4d_rope)
            USE_4D_ROPE=true
            shift
            ;;
        --no_4d_rope)
            USE_4D_ROPE=false
            shift
            ;;
        --rope_dim_ratios)
            ROPE_DIM_RATIOS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --fg_prompt)
            FG_PROMPT="$2"
            shift 2
            ;;
        --bg_prompt)
            BG_PROMPT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Layered Video Generation Inference"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "LoRA path: $LORA_PATH"
echo "Use EMA: $USE_EMA"
echo "Layer offset mode: $LAYER_OFFSET_MODE"
echo "Mask mode: $MASK_MODE"
if [ -n "$MASK_VAE_PATH" ]; then
    echo "MaskVAE path: $MASK_VAE_PATH"
fi
if [ -n "$MASK_VAE_PROJ_PATH" ]; then
    echo "MaskVAE proj path: $MASK_VAE_PROJ_PATH"
fi
if [ -n "$MASK_VAE_LORA_PATH" ]; then
    echo "Mask VAE LoRA path: $MASK_VAE_LORA_PATH"
fi
echo "Use 4D RoPE: $USE_4D_ROPE"
if [ -n "$ROPE_DIM_RATIOS" ]; then
    echo "RoPE dim ratios: $ROPE_DIM_RATIOS"
fi
echo "Output dir: $OUTPUT_DIR"
echo "Prompt: $PROMPT"
echo "FG prompt: $FG_PROMPT"
echo "BG prompt: $BG_PROMPT"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Frames: $FRAMES"
echo "Steps: $STEPS"
echo "Seed: $SEED"
echo "Fps: $FPS"
echo "Device: $DEVICE"
echo "=========================================="

# Build command
CMD="python -m wan.layered_t2v \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --prompt \"$PROMPT\" \
    --fg_prompt \"$FG_PROMPT\" \
    --bg_prompt \"$BG_PROMPT\" \
    --mask_mode $MASK_MODE \
    --width $WIDTH \
    --height $HEIGHT \
    --frames $FRAMES \
    --steps $STEPS \
    --seed $SEED \
    --device $DEVICE \
    --fps $FPS"

# Add 4D RoPE flag
if [ "$USE_4D_ROPE" = true ]; then
    CMD="$CMD --use_4d_rope"
else
    CMD="$CMD --no_4d_rope"
fi

# Add rope_dim_ratios if provided
if [ -n "$ROPE_DIM_RATIOS" ]; then
    CMD="$CMD --rope_dim_ratios \"$ROPE_DIM_RATIOS\""
fi

# Add LoRA path if provided
if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora_path \"$LORA_PATH\""
fi

# Add MaskVAE paths if provided
if [ -n "$MASK_VAE_PATH" ]; then
    CMD="$CMD --mask_vae_path \"$MASK_VAE_PATH\""
fi
if [ -n "$MASK_VAE_PROJ_PATH" ]; then
    CMD="$CMD --mask_vae_proj_path \"$MASK_VAE_PROJ_PATH\""
fi
if [ -n "$MASK_VAE_LORA_PATH" ]; then
    CMD="$CMD --mask_vae_lora_path \"$MASK_VAE_LORA_PATH\""
fi

# Add EMA flag if enabled
if [ "$USE_EMA" = true ]; then
    CMD="$CMD --use_ema"
fi

# Run inference
eval $CMD

echo "Inference completed! Outputs saved to $OUTPUT_DIR"
