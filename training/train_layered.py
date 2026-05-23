# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
Training script for Layered Video Generation with LoRA fine-tuning.

Features:
- Multi-GPU DDP training via Accelerate
- Flow Matching with logit-normal time sampling
- EMA model for generation quality
- Gradient checkpointing for memory efficiency
- Wandb/TensorBoard logging
"""

import argparse
import gc
import logging
import math
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Accelerate for multi-GPU
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

# Config management
try:
    from omegaconf import OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

# EMA
try:
    from diffusers.training_utils import EMAModel

    HAS_EMA = True
except ImportError:
    HAS_EMA = False
    print("Warning: diffusers.training_utils.EMAModel not available. EMA disabled.")

# Learning rate scheduler
from transformers import get_cosine_schedule_with_warmup

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import LayeredVideoDataset, collate_fn
from training.lora_utils import (
    apply_lora,
    get_lora_config,
    load_lora_weights,
    save_lora_weights,
    unfreeze_layer_adaln,
)
from training.logging_utils import (
    init_logging,
    log_metrics,
    finish_logging,
    get_trackers_list,
    save_wandb_info,
    WandbOfflineLogger,
    HAS_WANDB,
)
from wan.modules.layered_model import LayeredWanModel
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel

logger = get_logger(__name__)


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    config_args, remaining_args = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train Layered Video Generation",
        parents=[config_parser],
    )
    # Model
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to Wan2.1 checkpoint"
    )
    parser.add_argument(
        "--vae_path", type=str, default=None, help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "--t5_path", type=str, default=None, help="Path to T5 checkpoint"
    )
    parser.add_argument(
        "--t5_tokenizer", type=str, default="google/umt5-xxl", help="T5 tokenizer path"
    )

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=196, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=392, help="LoRA alpha")
    parser.add_argument(
        "--use_all_linear",
        dest="use_all_linear",
        action="store_true",
        help="Apply LoRA to all linear layers",
    )
    parser.add_argument(
        "--no_all_linear",
        dest="use_all_linear",
        action="store_false",
        help="Only apply LoRA to attention layers",
    )
    parser.set_defaults(use_all_linear=True)

    # Data
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset")
    parser.add_argument(
        "--jsonl_path", type=str, default=None, help="Path to training data"
    )
    parser.add_argument("--resolution_h", type=int, default=480, help="Video height")
    parser.add_argument("--resolution_w", type=int, default=720, help="Video width")
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of frames (4n+1)"
    )
    parser.add_argument("--fps", type=int, default=24, help="Target FPS")
    parser.add_argument(
        "--frame_sampling",
        type=str,
        default="continuous",
        choices=["uniform", "continuous"],
        help="Frame sampling strategy: 'uniform' (evenly spaced) or 'continuous' (random start + consecutive)",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=100)

    # EMA
    parser.add_argument(
        "--use_ema", dest="use_ema", action="store_true", help="Use EMA model"
    )
    parser.add_argument(
        "--no_ema", dest="use_ema", action="store_false", help="Disable EMA model"
    )
    parser.set_defaults(use_ema=False)
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # Memory optimization
    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing",
    )
    parser.set_defaults(gradient_checkpointing=False)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )

    # Logging
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--run_name", type=str, default="layered-video-lora")

    # Mask processing mode
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="vae",
        choices=[
            "vae",
            "downsample",
            "downsample-project",
            "vae-project",
            "vae-lora",
            "mask-vae-project",
            "mask-vae-joint",
        ],
        help="Mask processing mode: vae (encode through VAE), downsample (direct downsample + repeat channels), downsample-project (downsample + learnable projection), vae-project (VAE with learnable adapter), vae-lora (VAE + project-in + decoder LoRA), mask-vae-project (MaskVAE with learnable projection), mask-vae-joint (MaskVAE + projection joint training)",
    )
    parser.add_argument(
        "--mask_rec_loss",
        type=str,
        default="smoothl1",
        choices=["l1", "smoothl1"],
        help="Mask reconstruction loss type for downsample-project mode",
    )
    parser.add_argument(
        "--mask_rec_weight",
        type=float,
        default=0.1,
        help="Overall weight for mask reconstruction loss (downsample-project only)",
    )
    parser.add_argument(
        "--mask_grad_weight",
        type=float,
        default=0.3,
        help="Relative weight for mask gradient loss inside reconstruction loss",
    )
    parser.add_argument(
        "--mask_use_temporal_grad",
        action="store_true",
        help="Include temporal gradient loss for mask reconstruction",
    )

    parser.add_argument(
        "--mask_vae_path",
        type=str,
        default=None,
        help="Path to MaskVAE checkpoint (required for mask-vae-project, optional warm start for mask-vae-joint)",
    )
    parser.add_argument(
        "--mask_vae_lora_path",
        type=str,
        default=None,
        help="Path to MaskVAE LoRA checkpoint (required for vae-lora mode)",
    )
    parser.add_argument(
        "--mask_vae_proj_path",
        type=str,
        default=None,
        help="Path to pretrained projection layers (vae-project, mask-vae-project, or mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_hidden",
        type=int,
        default=96,
        help="MaskVAE hidden channels (mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_latent",
        type=int,
        default=16,
        help="MaskVAE latent channels (mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_res_blocks",
        type=int,
        default=2,
        help="MaskVAE residual blocks per stage (mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_mlp_ratio",
        type=int,
        default=4,
        help="MaskVAE bottleneck MLP ratio (mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_mlp_depth",
        type=int,
        default=1,
        help="MaskVAE bottleneck MLP depth (mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_proj_hidden",
        type=int,
        default=128,
        help="Hidden channels for projection layers",
    )
    parser.add_argument(
        "--mask_vae_proj_depth",
        type=int,
        default=2,
        help="MLP depth for projection layers",
    )
    parser.add_argument(
        "--mask_vae_proj_norm",
        type=str,
        default="rmsnorm",
        choices=["none", "rmsnorm"],
        help="Normalization type for projection layers (default: rmsnorm)",
    )
    parser.add_argument(
        "--mask_vae_rec_weight",
        type=float,
        default=0.1,
        help="Weight for mask reconstruction loss in mask-vae-project/mask-vae-joint mode",
    )
    parser.add_argument(
        "--mask_lora_rec_weight",
        type=float,
        default=0.0,
        help="Weight for mask reconstruction loss in vae-lora mode. Set > 0 to enable.",
    )

    # Layer separation constraints
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="Weight for reconstruction consistency loss (full = fg + bg*(1-mask)). Suggested: 0.1-0.5",
    )
    parser.add_argument(
        "--mutual_exclusivity_weight",
        type=float,
        default=0.0,
        help="Weight for mutual exclusivity loss (fg outside mask should be sparse). Suggested: 0.1-0.3",
    )

    # 4D RoPE
    parser.add_argument(
        "--use_4d_rope",
        dest="use_4d_rope",
        action="store_true",
        help="Use 4D RoPE (L, T, H, W) position encoding (default)",
    )
    parser.add_argument(
        "--no_4d_rope",
        dest="use_4d_rope",
        action="store_false",
        help="Use original 3D RoPE (T, H, W) position encoding",
    )
    parser.set_defaults(use_4d_rope=True)
    parser.add_argument(
        "--rope_dim_ratios",
        type=str,
        default=None,
        help="4D RoPE dimension allocation (L,T,H,W) as comma-separated ints, e.g. '8,42,40,38'. Must sum to head_dim (128).",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--no_val_split",
        action="store_true",
        default=True,
        help="Use all data for training (no validation split)",
    )

    # Wandb 配置
    parser.add_argument(
        "--use_wandb",
        dest="use_wandb",
        action="store_true",
        help="Enable Wandb logging (in addition to TensorBoard)",
    )
    parser.add_argument(
        "--no_wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Wandb logging",
    )
    parser.set_defaults(use_wandb=True)
    parser.add_argument(
        "--wandb_offline",
        dest="wandb_offline",
        action="store_true",
        help="Use Wandb offline mode (for GPU clusters without internet)",
    )
    parser.add_argument(
        "--wandb_online",
        dest="wandb_offline",
        action="store_false",
        help="Use Wandb online mode (requires internet)",
    )
    parser.set_defaults(wandb_offline=True)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="layert2v",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (team or username)",
    )

    # Load config file if provided (as defaults), then parse remaining CLI args as overrides.
    if config_args.config:
        if not HAS_OMEGACONF:
            parser.error(
                "`--config` requires omegaconf. Install with: pip install omegaconf"
            )
        config = OmegaConf.load(config_args.config)
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            parser.error(f"Config file must be a mapping/dict: {config_args.config}")

        # Handle _base_ inheritance
        if "_base_" in config_dict:
            base_path = config_dict.pop("_base_")
            config_dir = Path(config_args.config).parent
            base_full_path = config_dir / base_path
            if base_full_path.exists():
                base_config = OmegaConf.load(base_full_path)
                base_dict = OmegaConf.to_container(base_config, resolve=True)
                if isinstance(base_dict, dict):
                    base_dict.update(config_dict)
                    config_dict = base_dict

        parser.set_defaults(**config_dict)

    args = parser.parse_args(remaining_args)
    args.config = config_args.config

    if not args.model_path:
        parser.error("`--model_path` is required (or set `model_path` in `--config`).")
    if not args.data_root and not args.jsonl_path:
        # parser.error("`--data_root` is required (or set `data_root` in `--config`).")
        parser.error("`--data_root` or `--jsonl_path` is required.")

    return args


def generate_run_name(args) -> str:
    """
    Generate a meaningful run name based on key training parameters.

    Format: {lr}_{lora_rank}_{offset}_{mask}_{rope}_{timestamp}
    Example: lr1e-4_r64_offset-learn_mask-vae_rope4d_20241217_143052
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Learning rate (scientific notation)
    lr_str = f"lr{args.learning_rate:.0e}".replace("e-0", "e-")

    # LoRA rank
    lora_str = f"r{args.lora_rank}"

    # Mask mode
    mask_str = f"mask-{args.mask_mode}"

    # RoPE mode
    rope_str = "rope4d" if args.use_4d_rope else "rope3d"

    return f"{lr_str}_{lora_str}_{mask_str}_{rope_str}_{timestamp}"


def save_run_config(args, run_dir: Path):
    """Save training config to the run directory."""
    config_path = run_dir / "config.yaml"

    # Convert args to dict
    config_dict = vars(args).copy()

    # Convert Path objects to strings for YAML serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return config_path


def _mask_rec_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_type: str
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    return F.smooth_l1_loss(pred, target)


def _mask_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    use_temporal: bool,
) -> torch.Tensor:
    loss_fn = F.l1_loss if loss_type == "l1" else F.smooth_l1_loss
    losses = []

    # Spatial gradients
    if pred.size(3) > 1:
        pred_h = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        target_h = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        losses.append(loss_fn(pred_h, target_h))

    if pred.size(4) > 1:
        pred_w = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        target_w = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        losses.append(loss_fn(pred_w, target_w))

    if use_temporal and pred.size(2) > 1:
        pred_t = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        target_t = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        losses.append(loss_fn(pred_t, target_t))

    if not losses:
        return pred.new_tensor(0.0)

    return sum(losses) / len(losses)


def _downsample_mask_area(
    mask: torch.Tensor, target_size: tuple
) -> torch.Tensor:
    """
    Downsample mask using area mode for anti-aliasing.

    Args:
        mask: [B, 1, T, H, W] mask in [0, 1]
        target_size: (T', H', W') target latent size

    Returns:
        Downsampled mask [B, 1, T', H', W']
    """
    return F.interpolate(mask, size=target_size, mode="area")


def _compute_layer_constraints(
    v_pred: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
    T_prime: int,
    x0_full_gt: torch.Tensor,
    consistency_weight: float = 0.0,
    mutual_exclusivity_weight: float = 0.0,
) -> dict:
    """
    Compute layer separation constraint losses.

    Args:
        v_pred: [B, 16, 4*T', H', W'] predicted velocity
        noise: [B, 16, 4*T', H', W'] shared noise
        mask: [B, 1, T, H, W] GT mask in [0, 1]
        T_prime: latent time dimension
        x0_full_gt: [B, 16, T', H', W'] ground-truth full latent
        consistency_weight: weight for reconstruction consistency loss
        mutual_exclusivity_weight: weight for mutual exclusivity loss

    Returns:
        dict with loss values
    """
    result = {}

    if consistency_weight <= 0 and mutual_exclusivity_weight <= 0:
        return result

    # Recover x0_pred from v_pred: v = noise - x0, so x0 = noise - v
    x0_pred = noise - v_pred

    # Split components
    x0_bg = x0_pred[:, :, T_prime : 2 * T_prime]
    x0_fg = x0_pred[:, :, 2 * T_prime : 3 * T_prime]

    # Downsample GT mask to latent size using area mode
    latent_size = (T_prime, x0_full_gt.shape[3], x0_full_gt.shape[4])
    mask_down = _downsample_mask_area(mask, latent_size)  # [B, 1, T', H', W']
    # Expand to match latent channels
    mask_down = mask_down.expand(-1, x0_full_gt.shape[1], -1, -1, -1)  # [B, 16, T', H', W']

    # 1. Reconstruction Consistency Loss
    # Physical constraint: full = fg + bg * (1 - mask)
    if consistency_weight > 0:
        reconstructed = x0_fg + x0_bg * (1 - mask_down)
        # Anchor to GT full latent to avoid trivial self-consistency.
        loss_consistency = F.mse_loss(reconstructed, x0_full_gt)
        result["loss_consistency"] = loss_consistency
        result["loss_consistency_scaled"] = consistency_weight * loss_consistency

    # 2. Mutual Exclusivity Loss
    # Foreground should be empty (close to neutral) outside mask region
    # Background inside mask region should differ from full (occluded)
    if mutual_exclusivity_weight > 0:
        # Foreground outside mask should be minimal (close to 0 or background)
        # We encourage fg * (1 - mask) to be small
        fg_outside_mask = x0_fg * (1 - mask_down)
        # Use L1 to encourage sparsity
        loss_fg_outside = fg_outside_mask.abs().mean()

        # Alternatively: fg outside should match bg outside (both showing background)
        # bg_outside_mask = x0_bg * (1 - mask_down)
        # loss_fg_outside = F.mse_loss(fg_outside_mask, bg_outside_mask)

        result["loss_mutual_excl"] = loss_fg_outside
        result["loss_mutual_excl_scaled"] = mutual_exclusivity_weight * loss_fg_outside

    return result


def compute_loss(
    model,
    vae,
    batch,
    text_encoder,
    device,
    patch_size=(1, 2, 2),
    mask_mode="vae",
    mask_encoder=None,
    mask_decoder=None,
    mask_rec_weight=0.1,
    mask_grad_weight=0.3,
    mask_use_temporal_grad=False,
    mask_rec_loss="smoothl1",
    mask_vae=None,
    mask_vae_proj_in=None,
    mask_vae_proj_out=None,
    mask_vae_rec_weight=0.1,
    mask_lora_decoder=None,
    mask_lora_rec_weight=0.0,
    return_breakdown=False,
    # New layer constraint parameters
    consistency_weight=0.0,
    mutual_exclusivity_weight=0.0,
):
    """
    Compute Flow Matching loss.

    Flow Matching:
    - x_t = (1 - t) * x0 + t * noise
    - v_target = noise - x0
    - loss = MSE(v_pred, v_target)

    Time sampling: Logit-normal (SD3/Flux style)

    For vae-project mode:
    - Mask encoded through VAE (like vae mode)
    - After noise addition: project_in applied to mask slice of x_t
    - After model forward: project_out applied to mask slice of v_pred
    - No extra reconstruction loss needed
    For mask-vae-joint mode:
    - MaskVAE is trained end-to-end with proj_in/proj_out

    Args:
        mask_encoder: MaskEncoder layer for 'downsample-project' mode (optional)
        mask_decoder: MaskDecoder layer for 'downsample-project' mode (optional)
        mask_rec_weight: Overall weight for mask reconstruction loss
        mask_grad_weight: Relative weight for gradient loss inside reconstruction loss
        mask_use_temporal_grad: Include temporal gradient loss if True
        mask_rec_loss: Loss type for mask reconstruction ("l1" or "smoothl1")
        mask_lora_decoder: LoRA decoder for vae-lora mode reconstruction loss
        mask_lora_rec_weight: Weight for mask reconstruction loss in vae-lora mode
        return_breakdown: Return per-component loss breakdown if True
        consistency_weight: Weight for reconstruction consistency loss (full = fg + bg*(1-mask))
        mutual_exclusivity_weight: Weight for mutual exclusivity loss (fg outside mask should be sparse)
    """
    # Get batch data
    full_video = batch["full_video"].to(device)
    background = batch["background"].to(device)
    foreground = batch["foreground"].to(device)
    mask = batch["mask"].to(device)
    captions_full = batch["caption_full"]
    captions_fg = batch["caption_fg"]
    captions_bg = batch["caption_bg"]

    B = full_video.shape[0]

    # Encode text for LayeredCrossAttention
    with torch.no_grad():
        ctx_full = text_encoder(captions_full, device)
        ctx_fg = text_encoder(captions_fg, device)
        ctx_bg = text_encoder(captions_bg, device)

        context = [
            torch.cat([ctx_full[i], ctx_fg[i], ctx_bg[i]], dim=0) for i in range(B)
        ]
        prompt_lens = torch.tensor(
            [[len(ctx_full[i]), len(ctx_fg[i]), len(ctx_bg[i])] for i in range(B)],
            device=device, dtype=torch.long
        )

    # VAE encode all layers (frozen)
    with torch.no_grad():
        full_video_z = torch.stack(vae.encode([full_video[i] for i in range(B)]))
        background_z = torch.stack(vae.encode([background[i] for i in range(B)]))
        foreground_z = torch.stack(vae.encode([foreground[i] for i in range(B)]))

        # Mask encoding based on mode
        if mask_mode in ("vae", "vae-project", "vae-lora"):
            mask_3ch = mask.repeat(1, 3, 1, 1, 1) * 2 - 1
            mask_z = torch.stack(vae.encode([mask_3ch[i] for i in range(B)]))
        elif mask_mode == "downsample":
            latent_size = full_video_z.shape[2:]
            mask_down = F.interpolate(mask, size=latent_size, mode="trilinear", align_corners=False)
            mask_z = (mask_down * 2 - 1).repeat(1, 16, 1, 1, 1)
        elif mask_mode in ("mask-vae-project", "mask-vae-joint"):
            pass  # Handled outside no_grad block

    T_prime = full_video_z.shape[2]
    mask_rec_total = None
    mask_rec = None
    mask_grad = None

    # Mask post-processing based on mode
    mask_lora_rec = None
    mask_lora_rec_scaled = None
    if mask_mode == "vae-lora":
        if mask_vae_proj_in is None:
            raise RuntimeError("vae-lora mode requires mask_vae_proj_in")
        with torch.no_grad():
            mask_z = mask_vae_proj_in(mask_z)
    elif mask_mode == "downsample-project":
        mask_input = mask * 2 - 1
        latent_size = full_video_z.shape[2:]
        mask_z = mask_encoder(mask_input, target_size=latent_size)

        if mask_rec_weight > 0:
            if mask_decoder is None:
                raise RuntimeError("mask_rec_weight > 0 requires mask_decoder")
            mask_pred = mask_decoder(mask_z, target_size=mask_input.shape[2:])
            mask_rec = _mask_rec_loss(mask_pred, mask_input, mask_rec_loss)
            mask_grad = (
                _mask_grad_loss(mask_pred, mask_input, mask_rec_loss, mask_use_temporal_grad)
                if mask_grad_weight > 0 else mask_rec.new_tensor(0.0)
            )
            mask_rec_total = mask_rec + mask_grad_weight * mask_grad
    elif mask_mode in ("mask-vae-project", "mask-vae-joint"):
        if mask_vae is None or mask_vae_proj_in is None:
            raise RuntimeError(f"{mask_mode} mode requires mask_vae and mask_vae_proj_in")
        mask_input = mask * 2 - 1
        if mask_mode == "mask-vae-project":
            with torch.no_grad():
                mask_z_raw = mask_vae.encode(mask_input)
        else:
            mask_z_raw = mask_vae.encode(mask_input)
        mask_z = mask_vae_proj_in(mask_z_raw)

        if mask_vae_rec_weight > 0:
            if mask_vae_proj_out is None:
                raise RuntimeError(f"mask_vae_rec_weight > 0 requires mask_vae_proj_out")
            mask_z_proj_out = mask_vae_proj_out(mask_z)
            mask_pred = mask_vae.decode(mask_z_proj_out, target_shape=mask_input.shape[2:])
            mask_rec = _mask_rec_loss(mask_pred, mask_input, mask_rec_loss)
            mask_grad = (
                _mask_grad_loss(mask_pred, mask_input, mask_rec_loss, mask_use_temporal_grad)
                if mask_grad_weight > 0 else mask_rec.new_tensor(0.0)
            )
            mask_rec_total = mask_rec + mask_grad_weight * mask_grad
    elif mask_mode not in ("vae", "downsample", "vae-project", "vae-lora"):
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    # Concatenate all layers along T dimension
    # [B, 16, 4*T', H', W']
    x0 = torch.cat([full_video_z, background_z, foreground_z, mask_z], dim=2)

    # Sample timesteps using logit-normal distribution (SD3/Flux style)
    # t = sigmoid(N(0, 1)) -> concentrates around t=0.5
    t = torch.sigmoid(torch.randn(B, device=device))

    # Sample noise (shared across all layers)
    noise = torch.randn_like(x0)

    # Create noisy sample: x_t = (1 - t) * x0 + t * noise
    t_expand = t.view(B, 1, 1, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * noise

    # Target velocity: v = noise - x0
    v_target = noise - x0

    # For vae-project mode: apply project_in to mask slice of x_t
    if mask_mode == "vae-project":
        if mask_vae_proj_in is None:
            raise RuntimeError("vae-project mode requires mask_vae_proj_in")
        # Split x_t, apply project_in to mask slice, then concatenate back
        x_t_mask = x_t[:, :, 3 * T_prime : 4 * T_prime]  # [B, 16, T', H', W']
        x_t_mask_proj = mask_vae_proj_in(x_t_mask)
        x_t = torch.cat(
            [
                x_t[:, :, 0 : 3 * T_prime],
                x_t_mask_proj,
            ],
            dim=2,
        )

    # Compute seq_len for model
    _, C, T_concat, H_prime, W_prime = x_t.shape
    seq_len = math.ceil(
        (H_prime * W_prime) / (patch_size[1] * patch_size[2]) * T_concat
    )

    # Model forward pass
    # Convert timestep to [0, 1000] scale
    timesteps = t * 1000

    # Prepare model input
    x_t_list = [x_t[i] for i in range(B)]

    # Forward through model
    v_pred_list = model(
        x_t_list,
        t=timesteps,
        context=context,
        seq_len=seq_len,
        prompt_lens=prompt_lens,  # For LayeredCrossAttention
    )

    # Stack predictions
    v_pred = torch.stack(v_pred_list)  # [B, 16, 4*T', H', W']

    # For vae-project mode: apply project_out to mask slice of v_pred
    if mask_mode == "vae-project":
        if mask_vae_proj_out is None:
            raise RuntimeError("vae-project mode requires mask_vae_proj_out")
        # Split v_pred, apply project_out to mask slice, then concatenate back
        v_pred_mask = v_pred[:, :, 3 * T_prime : 4 * T_prime]  # [B, 16, T', H', W']
        v_pred_mask_proj = mask_vae_proj_out(v_pred_mask)
        v_pred = torch.cat(
            [
                v_pred[:, :, 0 : 3 * T_prime],
                v_pred_mask_proj,
            ],
            dim=2,
        )

    # Compute MSE loss
    loss_fm = F.mse_loss(v_pred, v_target)
    loss = loss_fm
    mask_rec_scaled = None

    # VAE-LoRA mask reconstruction loss (uses predicted mask latent)
    if mask_mode == "vae-lora" and mask_lora_rec_weight > 0:
        if mask_lora_decoder is None:
            raise RuntimeError("mask_lora_rec_weight > 0 requires mask_lora_decoder")
        mask_input = mask * 2 - 1  # [B, 1, T, H, W] in [-1, 1]
        mask_pred_latent = (
            noise[:, :, 3 * T_prime : 4 * T_prime]
            - v_pred[:, :, 3 * T_prime : 4 * T_prime]
        )
        # Decode projected latent back to pixel space
        # mask_lora_decoder.decode expects a list of [C, T, H, W] tensors
        mask_pred_list = mask_lora_decoder.decode(
            [mask_pred_latent[i] for i in range(mask_pred_latent.shape[0])]
        )
        # Average 3 channels to single channel, stack batch
        # Each item in mask_pred_list is [3, T, H, W]
        mask_pred = torch.stack([
            m.mean(dim=0, keepdim=True) for m in mask_pred_list
        ])  # [B, 1, T, H, W]
        mask_lora_rec = _mask_rec_loss(mask_pred, mask_input, mask_rec_loss)
        mask_lora_rec_scaled = mask_lora_rec * mask_lora_rec_weight

    if mask_rec_total is not None:
        # Use correct weight based on mode
        if mask_mode in ("mask-vae-project", "mask-vae-joint"):
            mask_rec_scaled = mask_vae_rec_weight * mask_rec_total
            loss = loss + mask_rec_scaled
        else:
            mask_rec_scaled = mask_rec_weight * mask_rec_total
            loss = loss + mask_rec_scaled

    # Add vae-lora reconstruction loss
    if mask_lora_rec_scaled is not None:
        loss = loss + mask_lora_rec_scaled

    # Compute layer separation constraints
    constraint_losses = _compute_layer_constraints(
        v_pred=v_pred,
        noise=noise,
        mask=mask,
        T_prime=T_prime,
        x0_full_gt=full_video_z,
        consistency_weight=consistency_weight,
        mutual_exclusivity_weight=mutual_exclusivity_weight,
    )
    if "loss_consistency_scaled" in constraint_losses:
        loss = loss + constraint_losses["loss_consistency_scaled"]
    if "loss_mutual_excl_scaled" in constraint_losses:
        loss = loss + constraint_losses["loss_mutual_excl_scaled"]

    if not return_breakdown:
        return loss

    # Compute per-layer losses for breakdown
    layer_losses = [
        F.mse_loss(v_pred[:, :, i*T_prime:(i+1)*T_prime], v_target[:, :, i*T_prime:(i+1)*T_prime])
        for i in range(4)
    ]

    loss_dict = {
        "loss_total": loss.detach(),
        "loss_fm": loss_fm.detach(),
        "loss_full_video": layer_losses[0].detach(),
        "loss_background": layer_losses[1].detach(),
        "loss_foreground": layer_losses[2].detach(),
        "loss_mask_latent": layer_losses[3].detach(),
        "mask_input_mean": mask.mean().detach(),
        "mask_input_min": mask.min().detach(),
        "mask_input_max": mask.max().detach(),
    }
    if mask_rec is not None:
        loss_dict["loss_mask_rec"] = mask_rec.detach()
    if mask_grad is not None:
        loss_dict["loss_mask_grad"] = mask_grad.detach()
    if mask_rec_total is not None:
        loss_dict["loss_mask_rec_total"] = mask_rec_total.detach()
    if mask_rec_scaled is not None:
        loss_dict["loss_mask_rec_scaled"] = mask_rec_scaled.detach()

    # Add vae-lora reconstruction loss to breakdown
    if mask_lora_rec is not None:
        loss_dict["loss_mask_lora_rec"] = mask_lora_rec.detach()
    if mask_lora_rec_scaled is not None:
        loss_dict["loss_mask_lora_rec_scaled"] = mask_lora_rec_scaled.detach()

    # Add layer constraint losses to breakdown
    if "loss_consistency" in constraint_losses:
        loss_dict["loss_consistency"] = constraint_losses["loss_consistency"].detach()
        loss_dict["loss_consistency_scaled"] = constraint_losses["loss_consistency_scaled"].detach()
    if "loss_mutual_excl" in constraint_losses:
        loss_dict["loss_mutual_excl"] = constraint_losses["loss_mutual_excl"].detach()
        loss_dict["loss_mutual_excl_scaled"] = constraint_losses["loss_mutual_excl_scaled"].detach()

    loss_dict["mask_input_mean"] = mask.mean().detach()
    loss_dict["mask_input_min"] = mask.min().detach()
    loss_dict["mask_input_max"] = mask.max().detach()

    return loss, loss_dict


def validate(
    model,
    vae,
    dataloader,
    text_encoder,
    device,
    patch_size=(1, 2, 2),
    mask_mode="vae",
    mask_encoder=None,
    mask_decoder=None,
    mask_rec_weight=0.1,
    mask_grad_weight=0.3,
    mask_use_temporal_grad=False,
    mask_rec_loss="smoothl1",
    accelerator=None,
    mask_vae=None,
    mask_vae_proj_in=None,
    mask_vae_proj_out=None,
    mask_vae_rec_weight=0.1,
    mask_lora_decoder=None,
    mask_lora_rec_weight=0.0,
    consistency_weight=0.0,
    mutual_exclusivity_weight=0.0,
):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if accelerator is not None:
                with accelerator.autocast():
                    loss = compute_loss(
                        model,
                        vae,
                        batch,
                        text_encoder,
                        device,
                        patch_size,
                        mask_mode,
                        mask_encoder,
                        mask_decoder,
                        mask_rec_weight,
                        mask_grad_weight,
                        mask_use_temporal_grad,
                        mask_rec_loss,
                        mask_vae,
                        mask_vae_proj_in,
                        mask_vae_proj_out,
                        mask_vae_rec_weight,
                        mask_lora_decoder=mask_lora_decoder,
                        mask_lora_rec_weight=mask_lora_rec_weight,
                        consistency_weight=consistency_weight,
                        mutual_exclusivity_weight=mutual_exclusivity_weight,
                    )
            else:
                loss = compute_loss(
                    model,
                    vae,
                    batch,
                    text_encoder,
                    device,
                    patch_size,
                    mask_mode,
                    mask_encoder,
                    mask_decoder,
                    mask_rec_weight,
                    mask_grad_weight,
                    mask_use_temporal_grad,
                    mask_rec_loss,
                    mask_vae,
                    mask_vae_proj_in,
                    mask_vae_proj_out,
                    mask_vae_rec_weight,
                    mask_lora_decoder=mask_lora_decoder,
                    mask_lora_rec_weight=mask_lora_rec_weight,
                    consistency_weight=consistency_weight,
                    mutual_exclusivity_weight=mutual_exclusivity_weight,
                )
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    model.train()
    return avg_loss


def save_checkpoint(
    accelerator,
    model,
    ema_model,
    optimizer,
    lr_scheduler,
    global_step,
    name,
    mask_encoder=None,
    mask_decoder=None,
    mask_vae=None,
    mask_vae_proj_in=None,
    mask_vae_proj_out=None,
    mask_vae_lora_state=None,
):
    """Save training checkpoint."""
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_dir = (
            Path(accelerator.project_configuration.project_dir) / "checkpoints" / name
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        unwrapped_model = accelerator.unwrap_model(model)
        save_lora_weights(unwrapped_model, str(save_dir))

        # Save mask_encoder/decoder separately (like VAE but trainable)
        # Unwrap DDP if needed
        if mask_encoder is not None:
            enc_to_save = (
                mask_encoder.module if hasattr(mask_encoder, "module") else mask_encoder
            )
            torch.save(enc_to_save.state_dict(), save_dir / "mask_encoder.pt")
        if mask_decoder is not None:
            dec_to_save = (
                mask_decoder.module if hasattr(mask_decoder, "module") else mask_decoder
            )
            torch.save(dec_to_save.state_dict(), save_dir / "mask_decoder.pt")

        # Save MaskVAE for joint training
        if mask_vae is not None:
            mask_vae_to_save = (
                mask_vae.module if hasattr(mask_vae, "module") else mask_vae
            )
            if any(p.requires_grad for p in mask_vae_to_save.parameters()):
                from wan.modules.mask_vae import save_mask_vae

                save_mask_vae(mask_vae_to_save, str(save_dir / "mask_vae.pt"))

        # Save mask_vae proj layers (mask-vae-project/mask-vae-joint mode)
        if mask_vae_proj_in is not None and mask_vae_proj_out is not None:
            proj_in_to_save = (
                mask_vae_proj_in.module
                if hasattr(mask_vae_proj_in, "module")
                else mask_vae_proj_in
            )
            proj_out_to_save = (
                mask_vae_proj_out.module
                if hasattr(mask_vae_proj_out, "module")
                else mask_vae_proj_out
            )
            torch.save(
                {
                    "proj_in": proj_in_to_save.state_dict(),
                    "proj_out": proj_out_to_save.state_dict(),
                    "config": {
                        "hidden_channels": proj_in_to_save.hidden_channels,
                        "mlp_depth": proj_in_to_save.mlp_depth,
                        "latent_channels": proj_in_to_save.latent_channels,
                        "norm_type": proj_in_to_save.norm_type,
                        "gate_init": proj_in_to_save.gate_init,
                    },
                },
                save_dir / "mask_vae_projects.pt",
            )

        if mask_vae_lora_state is not None:
            def _to_cpu(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu()
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                return obj

            torch.save(_to_cpu(mask_vae_lora_state), save_dir / "mask_vae_lora.pt")

        # Save optimizer and scheduler
        torch.save(
            {
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            save_dir / "training_state.pt",
        )

        # Save EMA model
        if ema_model is not None:
            ema_dir = save_dir / "ema"
            ema_dir.mkdir(exist_ok=True)
            torch.save(ema_model.state_dict(), ema_dir / "ema_model.pt")

        logger.info(f"Checkpoint saved to {save_dir}")


def load_checkpoint(
    accelerator,
    model,
    ema_model,
    optimizer,
    lr_scheduler,
    checkpoint_path,
    mask_encoder=None,
    mask_decoder=None,
    mask_vae=None,
    mask_vae_proj_in=None,
    mask_vae_proj_out=None,
):
    """Load training checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Load training state
    state_path = checkpoint_path / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        global_step = state["global_step"]
    else:
        global_step = 0

    # Load mask_encoder/decoder if saved separately
    # Note: load into underlying module if DDP-wrapped
    mask_enc_path = checkpoint_path / "mask_encoder.pt"
    if mask_encoder is not None and mask_enc_path.exists():
        enc_target = (
            mask_encoder.module if hasattr(mask_encoder, "module") else mask_encoder
        )
        enc_target.load_state_dict(torch.load(mask_enc_path, map_location="cpu"))
        enc_target.to(accelerator.device)
        logger.info(f"Loaded mask_encoder from {mask_enc_path}")

    mask_dec_path = checkpoint_path / "mask_decoder.pt"
    if mask_decoder is not None and mask_dec_path.exists():
        dec_target = (
            mask_decoder.module if hasattr(mask_decoder, "module") else mask_decoder
        )
        dec_target.load_state_dict(torch.load(mask_dec_path, map_location="cpu"))
        dec_target.to(accelerator.device)
        logger.info(f"Loaded mask_decoder from {mask_dec_path}")

    # Load MaskVAE for joint training
    mask_vae_path = checkpoint_path / "mask_vae.pt"
    if mask_vae is not None and mask_vae_path.exists():
        mask_vae_target = (
            mask_vae.module if hasattr(mask_vae, "module") else mask_vae
        )
        state = torch.load(mask_vae_path, map_location="cpu")
        mask_vae_target.encoder.load_state_dict(state["encoder"])
        mask_vae_target.decoder.load_state_dict(state["decoder"])
        mask_vae_target.to(accelerator.device)
        logger.info(f"Loaded MaskVAE from {mask_vae_path}")

    # Load mask_vae proj layers (mask-vae-project/mask-vae-joint mode)
    proj_path = checkpoint_path / "mask_vae_projects.pt"
    if (
        mask_vae_proj_in is not None
        and mask_vae_proj_out is not None
        and proj_path.exists()
    ):
        proj_state = torch.load(proj_path, map_location="cpu")
        proj_in_target = (
            mask_vae_proj_in.module
            if hasattr(mask_vae_proj_in, "module")
            else mask_vae_proj_in
        )
        proj_out_target = (
            mask_vae_proj_out.module
            if hasattr(mask_vae_proj_out, "module")
            else mask_vae_proj_out
        )
        proj_in_target.load_state_dict(proj_state["proj_in"])
        proj_out_target.load_state_dict(proj_state["proj_out"])
        proj_in_target.to(accelerator.device)
        proj_out_target.to(accelerator.device)
        logger.info(f"Loaded mask_vae_proj_in/out from {proj_path}")

    # Load EMA
    if ema_model is not None:
        ema_path = checkpoint_path / "ema" / "ema_model.pt"
        if ema_path.exists():
            ema_model.load_state_dict(torch.load(ema_path, map_location="cpu"))
            # Move EMA back to the correct device after loading
            ema_model.to(accelerator.device)

    logger.info(f"Checkpoint loaded from {checkpoint_path}, global_step={global_step}")
    return global_step


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Auto-generate run_name if using default value
    auto_generated_run_name = False
    if args.run_name == "layered-video-lora":
        args.run_name = generate_run_name(args)
        auto_generated_run_name = True

    # Initialize Accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )

    # Always enable find_unused_parameters for DDP
    # This is needed because not all parameters may be used in every forward pass
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # 获取日志记录器列表
    trackers = get_trackers_list(args.use_wandb, args.wandb_offline)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=trackers,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Log auto-generated run name (after accelerator init)
    if auto_generated_run_name:
        logger.info(f"Auto-generated run name: {args.run_name}")

    # 初始化 Wandb 离线日志记录器
    wandb_logger = None
    wandb_enabled = False
    from pathlib import Path
    if args.use_wandb and args.wandb_offline and accelerator.is_main_process:
        wandb_dir = Path(args.logging_dir) / "wandb" / args.run_name
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_logger = WandbOfflineLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            dir=str(wandb_dir),
            offline=True,
        )
        wandb_enabled = wandb_logger.init()
        if wandb_enabled:
            logger.info(f"Wandb 离线模式已启用")
            logger.info(f"  日志目录: {wandb_dir}")
            logger.info(f"  同步命令: wandb sync {wandb_dir}")
            save_wandb_info(args.logging_dir, args.run_name)

    # Initialize tracker (TensorBoard)
    # Logs will be saved to: logging_dir/run_name
    run_log_dir = Path(args.logging_dir) / args.run_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.run_name,
            config=vars(args),
        )
        # Save config.yaml to the run's log directory
        run_log_dir.mkdir(parents=True, exist_ok=True)
        config_path = save_run_config(args, run_log_dir)
        logger.info(f"TensorBoard logs: {run_log_dir}")
        logger.info(f"Config saved to: {config_path}")
        logger.info(f"  Run: tensorboard --logdir {args.logging_dir}")

    set_seed(args.seed)
    device = accelerator.device

    logger.info(f"Training on {accelerator.num_processes} GPUs")
    logger.info(f"Mixed precision: {args.mixed_precision}")

    # =====================
    # Load Models
    # =====================

    # Load VAE (frozen)
    logger.info("Loading VAE...")
    vae_path = args.vae_path or os.path.join(args.model_path, "Wan2.1_VAE.pth")
    vae = WanVAE(vae_pth=vae_path, device=device)
    vae.model.eval()
    for param in vae.model.parameters():
        param.requires_grad = False

    # Load T5 text encoder (frozen)
    logger.info("Loading T5 text encoder...")
    t5_path = args.t5_path or os.path.join(
        args.model_path, "models_t5_umt5-xxl-enc-bf16.pth"
    )
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=t5_path,
        tokenizer_path=args.t5_tokenizer,
    )
    text_encoder.model.eval()
    for param in text_encoder.model.parameters():
        param.requires_grad = False

    # Load DiT model and convert to LayeredWanModel
    logger.info("Loading DiT model...")
    from wan.modules.model import WanModel

    # Parse rope_dim_ratios if provided
    rope_dim_ratios = None
    if args.rope_dim_ratios:
        rope_dim_ratios = tuple(int(x) for x in args.rope_dim_ratios.split(","))
        if len(rope_dim_ratios) != 4:
            raise ValueError(
                f"rope_dim_ratios must have 4 values (L,T,H,W), got {len(rope_dim_ratios)}"
            )

    base_model = WanModel.from_pretrained(args.model_path)
    model = LayeredWanModel.from_pretrained_wan(
        base_model,
        num_output_layers=4,
        mask_mode=args.mask_mode,
        use_4d_rope=args.use_4d_rope,
        rope_dim_ratios=rope_dim_ratios,
    )
    del base_model

    # Extract mask_encoder/decoder BEFORE applying LoRA (managed separately like VAE)
    # This prevents DDP conflicts when they are called outside model.forward()
    mask_encoder = None
    mask_decoder = None
    mask_lora_decoder = None  # For vae-lora mode reconstruction loss
    mask_vae = None
    mask_vae_proj_in = None
    mask_vae_proj_out = None
    mask_vae_lora_state = None
    mask_vae_trainable = False
    if args.mask_mode == "downsample-project":
        mask_encoder = model.mask_encoder
        mask_decoder = model.mask_decoder
        # Remove from model to prevent PEFT/DDP wrapping
        model.mask_encoder = None
        model.mask_decoder = None
        logger.info("Extracted mask_encoder/decoder as standalone modules (like VAE)")
    elif args.mask_mode == "vae-project":
        # vae-project: use projection layers on noisy latent and predicted velocity
        # No adapter needed, just project_in/out
        from wan.modules.mask_vae_project import (
            MaskLatentProjectIn,
            MaskLatentProjectOut,
            load_mask_latent_projects,
        )

        if args.mask_vae_proj_path and os.path.exists(args.mask_vae_proj_path):
            mask_vae_proj_in, mask_vae_proj_out = load_mask_latent_projects(
                args.mask_vae_proj_path,
                device=device,
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            )
            logger.info(f"Loaded projection layers from {args.mask_vae_proj_path}")
        else:
            mask_vae_proj_in = MaskLatentProjectIn(
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            ).to(device)
            mask_vae_proj_out = MaskLatentProjectOut(
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            ).to(device)
            logger.info(
                f"Created new projection layers (norm={args.mask_vae_proj_norm})"
            )
        mask_vae_proj_in.train()
        mask_vae_proj_out.train()
    elif args.mask_mode == "vae-lora":
        from wan.modules.mask_vae_lora import load_mask_vae_lora_state, build_project_in_from_state

        mask_vae_lora_path = args.mask_vae_lora_path
        if not mask_vae_lora_path and args.resume_from:
            resume_candidate = Path(args.resume_from) / "mask_vae_lora.pt"
            if resume_candidate.exists():
                mask_vae_lora_path = str(resume_candidate)
                logger.info(
                    f"Using MaskVAE LoRA from resume checkpoint: {mask_vae_lora_path}"
                )

        if not mask_vae_lora_path:
            raise ValueError(
                "vae-lora mode requires --mask_vae_lora_path. "
                "Please run Stage 1 training (train_vae_lora.py) first, "
                "or pass --resume_from that contains mask_vae_lora.pt."
            )
        if not os.path.exists(mask_vae_lora_path):
            raise FileNotFoundError(
                f"mask_vae_lora_path not found: {mask_vae_lora_path}. "
                f"Please run Stage 1 training (train_vae_lora.py) first."
            )
        mask_vae_lora_state = load_mask_vae_lora_state(
            mask_vae_lora_path, device=device
        )
        mask_vae_proj_in = build_project_in_from_state(mask_vae_lora_state, device=device)
        logger.info(f"Loaded MaskVAE LoRA project-in from {mask_vae_lora_path}")

        # Load decoder for reconstruction loss if weight > 0
        if args.mask_lora_rec_weight > 0:
            from wan.modules.mask_vae_lora import build_decoder_from_state
            mask_lora_decoder = build_decoder_from_state(
                mask_vae_lora_state,
                vae_pth=vae_path,
                device=device,
                dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32,
            )
            logger.info(
                f"Loaded MaskVAE LoRA decoder for reconstruction loss (weight={args.mask_lora_rec_weight})"
            )
    elif args.mask_mode == "mask-vae-project":
        from wan.modules.mask_vae import MaskVAE, load_mask_vae
        from wan.modules.mask_vae_project import (
            MaskLatentProjectIn as MaskVAEProjectIn,
            MaskLatentProjectOut as MaskVAEProjectOut,
            load_mask_latent_projects as load_mask_vae_projects,
        )

        if not args.mask_vae_path:
            raise ValueError(
                "mask-vae-project mode requires --mask_vae_path. "
                "Please run Stage 1 training (train_maskvae.py) first."
            )
        if not os.path.exists(args.mask_vae_path):
            raise FileNotFoundError(
                f"mask_vae_path not found: {args.mask_vae_path}. "
                f"Please run Stage 1 training (train_maskvae.py) first."
            )
        mask_vae = load_mask_vae(args.mask_vae_path, device=device)
        mask_vae.eval()
        for p in mask_vae.parameters():
            p.requires_grad = False
        logger.info(f"Loaded and froze MaskVAE from {args.mask_vae_path}")

        if args.mask_vae_proj_path and os.path.exists(args.mask_vae_proj_path):
            mask_vae_proj_in, mask_vae_proj_out = load_mask_vae_projects(
                args.mask_vae_proj_path,
                device=device,
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            )
            logger.info(f"Loaded project layers from {args.mask_vae_proj_path}")
        else:
            mask_vae_proj_in = MaskVAEProjectIn(
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            ).to(device)
            mask_vae_proj_out = MaskVAEProjectOut(
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            ).to(device)
            logger.info("Created new project layers for training")
        mask_vae_proj_in.train()
        mask_vae_proj_out.train()
    elif args.mask_mode == "mask-vae-joint":
        from wan.modules.mask_vae import MaskVAE, load_mask_vae
        from wan.modules.mask_vae_project import (
            MaskLatentProjectIn as MaskVAEProjectIn,
            MaskLatentProjectOut as MaskVAEProjectOut,
            load_mask_latent_projects as load_mask_vae_projects,
        )

        if args.mask_vae_path:
            if not os.path.exists(args.mask_vae_path):
                raise FileNotFoundError(
                    f"mask_vae_path not found: {args.mask_vae_path}."
                )
            mask_vae = load_mask_vae(args.mask_vae_path, device=device)
            logger.info(f"Loaded MaskVAE from {args.mask_vae_path}")
        else:
            mask_vae = MaskVAE(
                hidden_channels=args.mask_vae_hidden,
                latent_channels=args.mask_vae_latent,
                num_res_blocks=args.mask_vae_res_blocks,
                mlp_ratio=args.mask_vae_mlp_ratio,
                mlp_depth=args.mask_vae_mlp_depth,
            ).to(device)
            logger.info("Created new MaskVAE for joint training")
        mask_vae.train()
        for p in mask_vae.parameters():
            p.requires_grad = True
        mask_vae_trainable = True

        if args.mask_vae_proj_path and os.path.exists(args.mask_vae_proj_path):
            mask_vae_proj_in, mask_vae_proj_out = load_mask_vae_projects(
                args.mask_vae_proj_path,
                device=device,
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            )
            logger.info(f"Loaded project layers from {args.mask_vae_proj_path}")
        else:
            mask_vae_proj_in = MaskVAEProjectIn(
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            ).to(device)
            mask_vae_proj_out = MaskVAEProjectOut(
                hidden_channels=args.mask_vae_proj_hidden,
                mlp_depth=args.mask_vae_proj_depth,
                norm_type=args.mask_vae_proj_norm,
            ).to(device)
            logger.info(
                f"Created new project layers for joint training (norm={args.mask_vae_proj_norm})"
            )
        mask_vae_proj_in.train()
        mask_vae_proj_out.train()

    if args.use_4d_rope:
        logger.info(
            f"Using 4D RoPE with dim_ratios: {rope_dim_ratios or '(8,42,40,38) default'}"
        )
    else:
        logger.info("Using original 3D RoPE")

    # Apply / load LoRA
    logger.info("Setting up LoRA...")
    if args.resume_from:
        # Load adapter weights first so optimizer sees the right params and values.
        logger.info(f"Loading LoRA weights from checkpoint: {args.resume_from}")
        for param in model.parameters():
            param.requires_grad = False
        model = load_lora_weights(model, args.resume_from, is_trainable=True)
        unfreeze_layer_adaln(model)
        # Note: mask_encoder/decoder are managed separately, not in PEFT
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        lora_config = get_lora_config(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            use_all_linear=args.use_all_linear,
            mask_mode=args.mask_mode,
        )
        model = apply_lora(model, lora_config)

    # Get patch_size from model (note: patch_size is in ignore_for_config, so not in config)
    # Navigate through PEFT wrapper if needed
    base = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model

    if hasattr(base, "patch_size"):
        patch_size = base.patch_size
    elif hasattr(base, "head") and hasattr(base.head, "patch_size"):
        patch_size = base.head.patch_size
    else:
        patch_size = (1, 2, 2)
        logger.warning(
            f"Could not get patch_size from model, using default: {patch_size}"
        )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        enabled = False
        # Navigate through PEFT wrapper to find the base model
        target = model
        if hasattr(target, "base_model") and hasattr(target.base_model, "model"):
            target = target.base_model.model

        # Try _set_gradient_checkpointing (our implementation) or enable_gradient_checkpointing (diffusers)
        if hasattr(target, "_set_gradient_checkpointing"):
            target._set_gradient_checkpointing(True)
            enabled = True
        elif hasattr(target, "enable_gradient_checkpointing"):
            target.enable_gradient_checkpointing()
            enabled = True

        if enabled:
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning(
                "`--gradient_checkpointing` was set, but the model does not implement gradient checkpointing."
            )

    model.train()

    # Collect trainable parameters once (LoRA + modules_to_save).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        # Best-effort fix for PEFT versions where `is_trainable` is not available.
        for name, param in model.named_parameters():
            if "lora_" in name or "modules_to_save" in name:
                param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found after applying/loading LoRA. "
            "Check `target_modules`, PEFT version, and that LoRA was attached successfully."
        )

    # Determine weight dtype from mixed_precision setting (more reliable than inspecting model params)
    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    mask_proj_trainable = args.mask_mode in (
        "vae-project",
        "mask-vae-project",
        "mask-vae-joint",
    )

    # Align MaskVAE dtype with model dtype to prevent fp32/bf16 mismatch
    if mask_vae is not None:
        mask_vae.to(device=device, dtype=weight_dtype)
        logger.info(f"MaskVAE dtype aligned to {weight_dtype}")

    # Setup mask_encoder/decoder as separate trainable modules (like VAE but trainable)
    # They will be wrapped by accelerator.prepare() for DDP/FSDP support
    mask_params = []
    if mask_vae is not None and mask_vae_trainable:
        mask_vae.train()
        for p in mask_vae.parameters():
            p.requires_grad = True
        mask_params.extend(list(mask_vae.parameters()))
        logger.info(f"mask_vae params: {sum(p.numel() for p in mask_vae.parameters())}")
    if mask_encoder is not None:
        mask_encoder.to(device=device, dtype=weight_dtype)
        mask_encoder.train()
        for p in mask_encoder.parameters():
            p.requires_grad = True
        mask_params.extend(list(mask_encoder.parameters()))
        logger.info(
            f"mask_encoder params: {sum(p.numel() for p in mask_encoder.parameters())}"
        )

    if mask_decoder is not None:
        mask_decoder.to(device=device, dtype=weight_dtype)
        mask_decoder.train()
        for p in mask_decoder.parameters():
            p.requires_grad = True
        mask_params.extend(list(mask_decoder.parameters()))
        logger.info(
            f"mask_decoder params: {sum(p.numel() for p in mask_decoder.parameters())}"
        )

    if mask_vae_proj_in is not None:
        mask_vae_proj_in.to(device=device, dtype=weight_dtype)
        if mask_proj_trainable:
            mask_vae_proj_in.train()
            for p in mask_vae_proj_in.parameters():
                p.requires_grad = True
            mask_params.extend(list(mask_vae_proj_in.parameters()))
            logger.info(
                f"mask_vae_proj_in params: {sum(p.numel() for p in mask_vae_proj_in.parameters())}"
            )

    if mask_vae_proj_out is not None:
        mask_vae_proj_out.to(device=device, dtype=weight_dtype)
        if mask_proj_trainable:
            mask_vae_proj_out.train()
            for p in mask_vae_proj_out.parameters():
                p.requires_grad = True
            mask_params.extend(list(mask_vae_proj_out.parameters()))
            logger.info(
                f"mask_vae_proj_out params: {sum(p.numel() for p in mask_vae_proj_out.parameters())}"
            )

    # Combine all trainable params for optimizer
    all_trainable_params = trainable_params + mask_params

    # Create EMA model
    ema_model = None
    if args.use_ema and HAS_EMA:
        logger.info("Creating EMA model...")
        ema_model = EMAModel(
            all_trainable_params,  # Include mask_encoder/decoder params
            decay=args.ema_decay,
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=0.75,
        )
        ema_model.to(device)

    # =====================
    # Data
    # =====================

    logger.info("Creating datasets...")
    train_dataset = LayeredVideoDataset(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        num_frames=args.num_frames,
        resolution=(args.resolution_h, args.resolution_w),
        fps=args.fps,
        split="train",
        no_val_split=args.no_val_split,
        frame_sampling=args.frame_sampling,
    )

    val_dataset = LayeredVideoDataset(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        num_frames=args.num_frames,
        resolution=(args.resolution_h, args.resolution_w),
        fps=args.fps,
        split="val",
        no_val_split=args.no_val_split,
        frame_sampling=args.frame_sampling,
    )

    # For small datasets, don't drop_last to avoid empty dataloaders
    # With DDP, each GPU gets len(dataset)/num_gpus samples
    drop_last = len(train_dataset) > args.batch_size * accelerator.num_processes

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # =====================
    # Optimizer & Scheduler
    # =====================

    optimizer = torch.optim.AdamW(
        all_trainable_params,  # Include mask_encoder/decoder params
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # Calculate total training steps
    # Note: With DDP, each GPU processes len(dataset)/num_gpus samples per epoch
    # So we need to divide by num_processes to get correct steps per epoch
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps
        / accelerator.num_processes
    )
    max_train_steps = (
        args.max_steps
        if args.max_steps > 0
        else args.num_epochs * num_update_steps_per_epoch
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    # =====================
    # Prepare with Accelerator
    # =====================

    # Prepare main model and training components
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Prepare mask modules separately (they are called outside model.forward())
    if mask_encoder is not None:
        mask_encoder = accelerator.prepare(mask_encoder)
        logger.info("Prepared mask_encoder with Accelerator")
    if mask_decoder is not None:
        mask_decoder = accelerator.prepare(mask_decoder)
        logger.info("Prepared mask_decoder with Accelerator")
    if mask_vae is not None and mask_vae_trainable:
        mask_vae = accelerator.prepare(mask_vae)
        logger.info("Prepared mask_vae with Accelerator")
    if mask_vae_proj_in is not None:
        mask_vae_proj_in = accelerator.prepare(mask_vae_proj_in)
        logger.info("Prepared mask_vae_proj_in with Accelerator")
    if mask_vae_proj_out is not None:
        mask_vae_proj_out = accelerator.prepare(mask_vae_proj_out)
        logger.info("Prepared mask_vae_proj_out with Accelerator")

    # =====================
    # Resume from checkpoint
    # =====================

    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(
            accelerator,
            model,
            ema_model,
            optimizer,
            lr_scheduler,
            args.resume_from,
            mask_encoder=mask_encoder,
            mask_decoder=mask_decoder,
            mask_vae=mask_vae,
            mask_vae_proj_in=mask_vae_proj_in,
            mask_vae_proj_out=mask_vae_proj_out,
        )

    # =====================
    # Training Loop
    # =====================

    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Num GPUs = {accelerator.num_processes}")
    logger.info(f"  Batch size per device = {args.batch_size}")
    logger.info(
        f"  Total batch size = {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Note: mask_encoder/decoder are already extracted and set up earlier (before LoRA)

    best_val_loss = float("inf")
    ema_loss = None  # EMA loss for smoother monitoring
    ema_decay = 0.99  # EMA decay factor

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
        desc="Training",
    )

    models_for_accum = [model]
    if mask_encoder is not None:
        models_for_accum.append(mask_encoder)
    if mask_decoder is not None:
        models_for_accum.append(mask_decoder)
    if mask_vae is not None and mask_vae_trainable:
        models_for_accum.append(mask_vae)
    if mask_vae_proj_in is not None:
        models_for_accum.append(mask_vae_proj_in)
    if mask_vae_proj_out is not None:
        models_for_accum.append(mask_vae_proj_out)

    for epoch in range(args.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*models_for_accum):
                # Compute loss
                with accelerator.autocast():
                    loss, loss_breakdown = compute_loss(
                        model,
                        vae,
                        batch,
                        text_encoder,
                        device,
                        patch_size,
                        args.mask_mode,
                        mask_encoder,
                        mask_decoder,
                        args.mask_rec_weight,
                        args.mask_grad_weight,
                        args.mask_use_temporal_grad,
                        args.mask_rec_loss,
                        mask_vae,
                        mask_vae_proj_in,
                        mask_vae_proj_out,
                        args.mask_vae_rec_weight,
                        mask_lora_decoder=mask_lora_decoder,
                        mask_lora_rec_weight=args.mask_lora_rec_weight,
                        return_breakdown=True,
                        consistency_weight=args.consistency_weight,
                        mutual_exclusivity_weight=args.mutual_exclusivity_weight,
                    )

                # Backward
                accelerator.backward(loss)

                # Gradient clipping (returns grad norm before clipping)
                # Use optimizer.param_groups for robustness with FSDP/DeepSpeed
                grad_norm = None
                if accelerator.sync_gradients:
                    params_to_clip = [
                        p for group in optimizer.param_groups for p in group["params"]
                    ]
                    grad_norm = accelerator.clip_grad_norm_(
                        params_to_clip, args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA
                if ema_model is not None and accelerator.sync_gradients:
                    ema_model.step(all_trainable_params)

            # Update progress
            if accelerator.sync_gradients:
                global_step += 1
                current_loss = loss.detach().item()
                current_lr = lr_scheduler.get_last_lr()[0]

                # Update EMA loss
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss

                # Update progress bar with current info (every step)
                progress_bar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    epoch=epoch,
                )
                progress_bar.update(1)

                # Logging to tensorboard (every log_steps)
                if global_step % args.log_steps == 0:
                    logs = {
                        "train/loss": current_loss,
                        "train/loss_ema": ema_loss,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                    }
                    if loss_breakdown is not None:
                        logs.update(
                            {
                                "train/loss_fm": loss_breakdown["loss_fm"].item(),
                                "train/loss_full_video": loss_breakdown[
                                    "loss_full_video"
                                ].item(),
                                "train/loss_background": loss_breakdown[
                                    "loss_background"
                                ].item(),
                                "train/loss_foreground": loss_breakdown[
                                    "loss_foreground"
                                ].item(),
                                "train/loss_mask_latent": loss_breakdown[
                                    "loss_mask_latent"
                                ].item(),
                                "train/mask_input_mean": loss_breakdown[
                                    "mask_input_mean"
                                ].item(),
                                "train/mask_input_min": loss_breakdown[
                                    "mask_input_min"
                                ].item(),
                                "train/mask_input_max": loss_breakdown[
                                    "mask_input_max"
                                ].item(),
                            }
                        )
                        if "loss_mask_rec" in loss_breakdown:
                            logs["train/loss_mask_rec"] = loss_breakdown[
                                "loss_mask_rec"
                            ].item()
                        if "loss_mask_grad" in loss_breakdown:
                            logs["train/loss_mask_grad"] = loss_breakdown[
                                "loss_mask_grad"
                            ].item()
                        if "loss_mask_rec_total" in loss_breakdown:
                            logs["train/loss_mask_rec_total"] = loss_breakdown[
                                "loss_mask_rec_total"
                            ].item()
                        if "loss_mask_rec_scaled" in loss_breakdown:
                            logs["train/loss_mask_rec_scaled"] = loss_breakdown[
                                "loss_mask_rec_scaled"
                            ].item()
                        # vae-lora mask reconstruction loss
                        if "loss_mask_lora_rec" in loss_breakdown:
                            logs["train/loss_mask_lora_rec"] = loss_breakdown[
                                "loss_mask_lora_rec"
                            ].item()
                        if "loss_mask_lora_rec_scaled" in loss_breakdown:
                            logs["train/loss_mask_lora_rec_scaled"] = loss_breakdown[
                                "loss_mask_lora_rec_scaled"
                            ].item()
                        # Layer separation constraint losses
                        if "loss_consistency" in loss_breakdown:
                            logs["train/loss_consistency"] = loss_breakdown[
                                "loss_consistency"
                            ].item()
                        if "loss_consistency_scaled" in loss_breakdown:
                            logs["train/loss_consistency_scaled"] = loss_breakdown[
                                "loss_consistency_scaled"
                            ].item()
                        if "loss_mutual_excl" in loss_breakdown:
                            logs["train/loss_mutual_excl"] = loss_breakdown[
                                "loss_mutual_excl"
                            ].item()
                        if "loss_mutual_excl_scaled" in loss_breakdown:
                            logs["train/loss_mutual_excl_scaled"] = loss_breakdown[
                                "loss_mutual_excl_scaled"
                            ].item()
                    # Add gradient norm if available
                    if grad_norm is not None:
                        logs["train/grad_norm"] = (
                            grad_norm.item()
                            if hasattr(grad_norm, "item")
                            else grad_norm
                        )
                    accelerator.log(logs, step=global_step)
                    # Wandb 离线日志
                    if wandb_logger is not None and wandb_enabled:
                        wandb_logger.log(logs, step=global_step)

                # Validation
                if global_step % args.validation_steps == 0 and len(val_dataset) > 0:
                    val_loss = validate(
                        model,
                        vae,
                        val_dataloader,
                        text_encoder,
                        device,
                        patch_size,
                        args.mask_mode,
                        mask_encoder,
                        mask_decoder,
                        args.mask_rec_weight,
                        args.mask_grad_weight,
                        args.mask_use_temporal_grad,
                        args.mask_rec_loss,
                        accelerator=accelerator,
                        mask_vae=mask_vae,
                        mask_vae_proj_in=mask_vae_proj_in,
                        mask_vae_proj_out=mask_vae_proj_out,
                        mask_vae_rec_weight=args.mask_vae_rec_weight,
                        mask_lora_decoder=mask_lora_decoder,
                        mask_lora_rec_weight=args.mask_lora_rec_weight,
                        consistency_weight=args.consistency_weight,
                        mutual_exclusivity_weight=args.mutual_exclusivity_weight,
                    )
                    accelerator.log({"val/loss": val_loss}, step=global_step)
                    # Wandb 验证日志
                    if wandb_logger is not None and wandb_enabled:
                        wandb_logger.log({"val/loss": val_loss}, step=global_step)
                    logger.info(f"Step {global_step}: val_loss = {val_loss:.6f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            accelerator,
                            model,
                            ema_model,
                            optimizer,
                            lr_scheduler,
                            global_step,
                            "best",
                            mask_encoder=mask_encoder,
                            mask_decoder=mask_decoder,
                            mask_vae=mask_vae,
                            mask_vae_proj_in=mask_vae_proj_in,
                            mask_vae_proj_out=mask_vae_proj_out,
                            mask_vae_lora_state=mask_vae_lora_state,
                        )

                # Save periodic checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        ema_model,
                        optimizer,
                        lr_scheduler,
                        global_step,
                        f"step-{global_step}",
                        mask_encoder=mask_encoder,
                        mask_decoder=mask_decoder,
                        mask_vae=mask_vae,
                        mask_vae_proj_in=mask_vae_proj_in,
                        mask_vae_proj_out=mask_vae_proj_out,
                        mask_vae_lora_state=mask_vae_lora_state,
                    )

                # Check max steps
                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # Save final checkpoint
    save_checkpoint(
        accelerator,
        model,
        ema_model,
        optimizer,
        lr_scheduler,
        global_step,
        "final",
        mask_encoder=mask_encoder,
        mask_decoder=mask_decoder,
        mask_vae=mask_vae,
        mask_vae_proj_in=mask_vae_proj_in,
        mask_vae_proj_out=mask_vae_proj_out,
        mask_vae_lora_state=mask_vae_lora_state,
    )

    accelerator.end_training()
    # 结束 Wandb 日志
    if wandb_logger is not None and wandb_enabled:
        wandb_logger.finish()
        logger.info("Wandb 日志已保存")
        wandb_dir = Path(args.logging_dir) / "wandb" / args.run_name
        logger.info(f"  运行 'wandb sync {wandb_dir}' 以同步到服务器")
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
