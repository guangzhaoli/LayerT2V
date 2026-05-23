#!/usr/bin/env python
# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
Stage 1: MaskVAE Pretraining Script.

Pretrain the MaskVAE to learn mask encoding/decoding with 16-channel latent space.
This MaskVAE is then frozen in Stage 2 (train_layered.py with mask-vae-project mode)
or used as an optional warm start for mask-vae-joint mode.

Usage:
    accelerate launch --config_file training/configs/accelerate_config.yaml \
        training/train_maskvae.py --config training/configs/default.yaml

    # Or single GPU:
    python training/train_maskvae.py --config training/configs/default.yaml
"""

import argparse
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from accelerate.logging import get_logger

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    from logging import getLogger as get_logger

try:
    from omegaconf import OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

from training.dataset import MaskOnlyDataset, collate_fn_mask_only
from training.logging_utils import (
    WandbOfflineLogger,
    get_trackers_list,
    save_wandb_info,
    HAS_WANDB,
)
from wan.modules.mask_vae import MaskVAE, MaskVAELoss, save_mask_vae

logger = get_logger(__name__)


def parse_args():
    # First pass: just get --config
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_args = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Stage 1: MaskVAE Pretraining",
        parents=[config_parser],
    )

    parser.add_argument(
        "--data_root", type=str, default=None, help="Path to training data"
    )

    parser.add_argument("--resolution_h", type=int, default=384)
    parser.add_argument("--resolution_w", type=int, default=672)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--frame_sampling",
        type=str,
        default="continuous",
        choices=["uniform", "continuous"],
    )

    parser.add_argument("--mask_vae_hidden", type=int, default=96)
    parser.add_argument("--mask_vae_latent", type=int, default=16)
    parser.add_argument("--mask_vae_res_blocks", type=int, default=2)
    parser.add_argument("--mask_vae_mlp_ratio", type=int, default=4)
    parser.add_argument("--mask_vae_mlp_depth", type=int, default=1)

    parser.add_argument("--grad_weight", type=float, default=0.2)
    parser.add_argument("--temporal_grad_weight", type=float, default=0.05)
    parser.add_argument("--edge_weight", type=float, default=0.1)
    parser.add_argument("--edge_scale", type=float, default=2.0)
    parser.add_argument(
        "--rec_loss_type", type=str, default="smoothl1", choices=["smoothl1", "l1"]
    )

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )

    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./outputs/mask_vae")
    parser.add_argument("--logging_dir", type=str, default="./logs/mask_vae")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume_from", type=str, default=None)

    # Wandb 配置
    parser.add_argument(
        "--use_wandb",
        dest="use_wandb",
        action="store_true",
        help="Enable Wandb logging",
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
        help="Use Wandb offline mode",
    )
    parser.add_argument(
        "--wandb_online",
        dest="wandb_offline",
        action="store_false",
        help="Use Wandb online mode",
    )
    parser.set_defaults(wandb_offline=True)
    parser.add_argument("--wandb_project", type=str, default="layert2v-maskvae")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Load config file BEFORE parsing remaining args (so CLI overrides config)
    if config_args.config and HAS_OMEGACONF:
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

    return args


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    mask_vae, optimizer, lr_scheduler, global_step, output_dir, name="checkpoint"
):
    save_path = os.path.join(output_dir, name)
    os.makedirs(save_path, exist_ok=True)

    save_mask_vae(mask_vae, os.path.join(save_path, "mask_vae.pt"))

    training_state = {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(training_state, os.path.join(save_path, "training_state.pt"))

    logger.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(mask_vae, optimizer, lr_scheduler, checkpoint_path, device):
    from wan.modules.mask_vae import load_mask_vae

    vae_path = os.path.join(checkpoint_path, "mask_vae.pt")
    if os.path.exists(vae_path):
        loaded_vae = load_mask_vae(vae_path, device=device)
        mask_vae.load_state_dict(loaded_vae.state_dict())
        logger.info(f"Loaded MaskVAE from {vae_path}")

    state_path = os.path.join(checkpoint_path, "training_state.pt")
    global_step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        global_step = state.get("global_step", 0)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info(f"Resumed from step {global_step}")

    return global_step


def main():
    args = parse_args()

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"mask_vae_lr{args.learning_rate}_{timestamp}"

    if HAS_ACCELERATE:
        trackers = get_trackers_list(args.use_wandb, args.wandb_offline)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=trackers,
            project_dir=args.logging_dir,
        )
        device = accelerator.device
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator is None or accelerator.is_main_process:
        if HAS_ACCELERATE:
            set_seed(args.seed)
        else:
            torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Stage 1: MaskVAE Pretraining")
    logger.info("=" * 60)
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Run name: {args.run_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # 初始化 Wandb 离线日志
    wandb_logger = None
    wandb_enabled = False
    if args.use_wandb and args.wandb_offline:
        is_main = accelerator is None or accelerator.is_main_process
        if is_main:
            from pathlib import Path
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
                logger.info(f"Wandb 离线模式已启用: {wandb_dir}")
                save_wandb_info(args.logging_dir, args.run_name)

    logger.info("Creating MaskVAE...")
    mask_vae = MaskVAE(
        hidden_channels=args.mask_vae_hidden,
        latent_channels=args.mask_vae_latent,
        num_res_blocks=args.mask_vae_res_blocks,
        mlp_ratio=args.mask_vae_mlp_ratio,
        mlp_depth=args.mask_vae_mlp_depth,
    )
    mask_vae.to(device)
    mask_vae.train()

    trainable_params = sum(p.numel() for p in mask_vae.parameters() if p.requires_grad)
    logger.info(f"MaskVAE trainable parameters: {trainable_params:,}")

    loss_fn = MaskVAELoss(
        rec_loss_type=args.rec_loss_type,
        grad_weight=args.grad_weight,
        temporal_grad_weight=args.temporal_grad_weight,
        edge_weight=args.edge_weight,
        edge_scale=args.edge_scale,
    )
    loss_fn.to(device)

    logger.info("Loading dataset...")
    train_dataset = MaskOnlyDataset(
        data_root=args.data_root,
        num_frames=args.num_frames,
        resolution=(args.resolution_h, args.resolution_w),
        fps=args.fps,
        frame_sampling=args.frame_sampling,
        split="train",
        no_val_split=True,
    )
    logger.info(f"Dataset size: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_mask_only,
    )

    optimizer = torch.optim.AdamW(
        mask_vae.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_steps > 0:
        max_train_steps = args.max_steps
    else:
        max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    if accelerator is not None:
        mask_vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            mask_vae, optimizer, train_dataloader, lr_scheduler
        )
        accelerator.init_trackers(
            project_name=args.run_name,
            config=vars(args),
        )

    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(
            mask_vae, optimizer, lr_scheduler, args.resume_from, device
        )

    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps: {max_train_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Grad weight: {args.grad_weight}")
    logger.info(f"  Temporal grad weight: {args.temporal_grad_weight}")
    logger.info(f"  Edge weight: {args.edge_weight}")
    logger.info("=" * 60)

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=accelerator is not None and not accelerator.is_main_process,
        desc="Training",
    )

    ema_loss = None
    ema_decay = 0.99

    for epoch in range(args.num_epochs):
        mask_vae.train()

        for step, batch in enumerate(train_dataloader):
            if accelerator is not None:
                with accelerator.accumulate(mask_vae):
                    mask = batch["mask"].to(device)
                    mask_input = mask * 2 - 1

                    with accelerator.autocast():
                        recon_mask, _ = mask_vae(mask_input, return_latent=True)
                        loss, loss_dict = loss_fn(recon_mask, mask_input)

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            mask_vae.parameters(), args.max_grad_norm
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                mask = batch["mask"].to(device)
                mask_input = mask * 2 - 1

                recon_mask, _ = mask_vae(mask_input, return_latent=True)
                loss, loss_dict = loss_fn(recon_mask, mask_input)

                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        mask_vae.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            sync_gradients = (
                accelerator.sync_gradients
                if accelerator
                else (step + 1) % args.gradient_accumulation_steps == 0
            )

            if sync_gradients:
                global_step += 1
                current_loss = loss_dict["loss_total"]
                current_lr = lr_scheduler.get_last_lr()[0]

                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss

                progress_bar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    ema=f"{ema_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    epoch=epoch,
                )
                progress_bar.update(1)

                if global_step % args.log_steps == 0:
                    logs = {
                        "train/loss": current_loss,
                        "train/loss_ema": ema_loss,
                        "train/loss_rec": loss_dict["loss_rec"],
                        "train/loss_grad": loss_dict["loss_grad"],
                        "train/loss_temp_grad": loss_dict["loss_temp_grad"],
                        "train/loss_edge": loss_dict["loss_edge"],
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                    }
                    # 可选指标
                    if "loss_perceptual" in loss_dict:
                        logs["train/loss_perceptual"] = loss_dict["loss_perceptual"]
                    if accelerator is not None:
                        accelerator.log(logs, step=global_step)
                    # Wandb 离线日志
                    if wandb_logger is not None and wandb_enabled:
                        wandb_logger.log(logs, step=global_step)

                if global_step % args.save_steps == 0:
                    is_main = accelerator is None or accelerator.is_main_process
                    if is_main:
                        unwrapped_vae = (
                            accelerator.unwrap_model(mask_vae)
                            if accelerator
                            else mask_vae
                        )
                        save_checkpoint(
                            unwrapped_vae,
                            optimizer,
                            lr_scheduler,
                            global_step,
                            os.path.join(args.output_dir, "checkpoints"),
                            f"step-{global_step}",
                        )

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    is_main = accelerator is None or accelerator.is_main_process
    if is_main:
        unwrapped_vae = accelerator.unwrap_model(mask_vae) if accelerator else mask_vae
        save_checkpoint(
            unwrapped_vae,
            optimizer,
            lr_scheduler,
            global_step,
            os.path.join(args.output_dir, "checkpoints"),
            "final",
        )
        final_vae_path = os.path.join(args.output_dir, "mask_vae.pt")
        save_mask_vae(unwrapped_vae, final_vae_path)
        logger.info(f"Final MaskVAE saved to {final_vae_path}")

    if accelerator is not None:
        accelerator.end_training()

    # 结束 Wandb 日志
    if wandb_logger is not None and wandb_enabled:
        wandb_logger.finish()
        logger.info("Wandb 日志已保存")

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Final MaskVAE: {os.path.join(args.output_dir, 'mask_vae.pt')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
