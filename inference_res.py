#!/usr/bin/env python3
# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import decord

    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

sys.path.insert(0, str(Path(__file__).parent))
from wan.modules.vae import WanVAE
from wan.modules.mask_adapter import MaskVAEAdapter, load_mask_adapter
from wan.modules.mask_vae import MaskVAE, load_mask_vae
from wan.modules.mask_vae_lora import (
    MaskVAELoRA,
    MaskVAELoRAConfig,
    load_mask_vae_lora_state,
)


def load_video(path: str, num_frames: int = None) -> torch.Tensor:
    """Load video as tensor [C, T, H, W] in range [-1, 1]."""
    if not HAS_DECORD:
        raise ImportError("decord is required for video loading")

    vr = decord.VideoReader(str(path))
    total_frames = len(vr)

    if num_frames is None or num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = list(range(num_frames))

    frames = vr.get_batch(indices)  # [T, H, W, C]
    frames = frames.permute(3, 0, 1, 2).float()  # [C, T, H, W]
    frames = frames / 127.5 - 1.0  # normalize to [-1, 1]
    return frames


def load_mask_video(path: str, num_frames: int = None) -> torch.Tensor:
    """Load mask video as tensor [1, T, H, W] in range [-1, 1]."""
    if not HAS_DECORD:
        raise ImportError("decord is required for video loading")

    vr = decord.VideoReader(str(path))
    total_frames = len(vr)

    if num_frames is None or num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = list(range(num_frames))

    frames = vr.get_batch(indices)  # [T, H, W, C]
    frames = frames.float().mean(dim=-1, keepdim=True)  # grayscale [T, H, W, 1]
    frames = frames.permute(3, 0, 1, 2)  # [1, T, H, W]
    frames = frames / 127.5 - 1.0  # normalize to [-1, 1]
    return frames


def resize_video(video: torch.Tensor, size: tuple) -> torch.Tensor:
    """Resize video [C, T, H, W] to target size (H, W)."""
    C, T, H, W = video.shape
    video_2d = video.permute(1, 0, 2, 3)  # [T, C, H, W]
    video_resized = F.interpolate(
        video_2d, size=size, mode="bilinear", align_corners=False
    )
    return video_resized.permute(1, 0, 2, 3)  # [C, T, H, W]


def tensor_to_video_frames(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor [C, T, H, W] range [-1, 1] to numpy [T, H, W, C] range [0, 255]."""
    tensor = (tensor.clamp(-1, 1) + 1) * 127.5
    tensor = tensor.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    return tensor


def save_video(frames: np.ndarray, path: str, fps: int = 8):
    """Save numpy frames [T, H, W, C] as video."""
    if not HAS_IMAGEIO:
        raise ImportError("imageio is required for video saving")
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def save_comparison_grid(
    original: np.ndarray, reconstructed: np.ndarray, path: str, fps: int = 8
):
    """Save side-by-side comparison video."""
    T, H, W, C = original.shape
    grid = np.zeros((T, H, W * 2 + 10, C), dtype=np.uint8)
    grid[:, :, :W, :] = original
    grid[:, :, W + 10 :, :] = reconstructed
    save_video(grid, path, fps)


def save_frame_grid(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    path: str,
    frame_indices: list = None,
):
    """Save comparison grid image for selected frames."""
    if not HAS_PIL:
        raise ImportError("PIL is required for image saving")

    C, T, H, W = original.shape

    if frame_indices is None:
        frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        frame_indices = [min(i, T - 1) for i in frame_indices]

    num_frames = len(frame_indices)
    grid_h = 2
    grid_w = num_frames

    grid = np.zeros(
        (H * grid_h + 10, W * grid_w + 10 * (grid_w - 1), 3), dtype=np.uint8
    )

    for col, frame_idx in enumerate(frame_indices):
        orig_frame = tensor_to_video_frames(
            original[:, frame_idx : frame_idx + 1, :, :]
        )[0]
        recon_frame = tensor_to_video_frames(
            reconstructed[:, frame_idx : frame_idx + 1, :, :]
        )[0]

        x_offset = col * (W + 10)

        if orig_frame.shape[-1] == 1:
            orig_frame = np.repeat(orig_frame, 3, axis=-1)
        if recon_frame.shape[-1] == 1:
            recon_frame = np.repeat(recon_frame, 3, axis=-1)

        grid[:H, x_offset : x_offset + W, :] = orig_frame
        grid[H + 10 :, x_offset : x_offset + W, :] = recon_frame

    Image.fromarray(grid).save(path)


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute reconstruction metrics."""
    mse = F.mse_loss(reconstructed, original).item()
    psnr = (
        10 * np.log10(4.0 / mse) if mse > 0 else float("inf")
    )  # range [-1,1] so max diff is 2
    mae = F.l1_loss(reconstructed, original).item()

    return {
        "mse": mse,
        "psnr": psnr,
        "mae": mae,
    }


def reconstruct_with_raw_vae(
    mask: torch.Tensor, vae, dtype: torch.dtype
) -> torch.Tensor:
    """Reconstruct mask using raw VAE (1ch -> repeat 3ch -> VAE -> mean -> 1ch)."""
    B, C, T, H, W = mask.shape
    mask_3ch = mask.repeat(1, 3, 1, 1, 1)  # [B, 3, T, H, W]

    with torch.amp.autocast("cuda", dtype=dtype):
        latent_list = vae.encode([x for x in mask_3ch])
        decoded_list = vae.decode(latent_list)
        decoded = torch.stack(decoded_list)  # [B, 3, T, H, W]

    recon_mask = decoded.mean(dim=1, keepdim=True)  # [B, 1, T, H, W]
    return recon_mask


def reconstruct_with_maskvae(
    mask: torch.Tensor, maskvae: MaskVAE, dtype: torch.dtype
) -> tuple:
    """
    Reconstruct mask using standalone MaskVAE (stage1).

    Args:
        mask: [B, 1, T, H, W] in range [-1, 1]
        maskvae: MaskVAE model
        dtype: compute dtype

    Returns:
        recon_mask: [B, 1, T, H, W] reconstructed mask
        latent: [B, 16, T', H', W'] latent representation
    """
    with torch.amp.autocast("cuda", dtype=dtype):
        recon_mask, latent = maskvae(mask, return_latent=True)
    return recon_mask, latent


def reconstruct_with_vae_lora(
    mask: torch.Tensor, vae_lora: MaskVAELoRA, dtype: torch.dtype
) -> tuple:
    """
    Reconstruct mask using MaskVAELoRA (vae-lora mode stage1).

    Args:
        mask: [B, 1, T, H, W] in range [0, 1]
        vae_lora: MaskVAELoRA model
        dtype: compute dtype

    Returns:
        recon_mask: [B, 1, T, H, W] reconstructed mask in range [-1, 1]
        latent: [B, 16, T', H', W'] latent representation
    """
    with torch.amp.autocast("cuda", dtype=dtype):
        # MaskVAELoRA expects mask in [0, 1], internally converts to [-1, 1]
        # Input mask is in [-1, 1], convert to [0, 1]
        mask_01 = (mask + 1) / 2
        recon_mask = vae_lora(mask_01)
        # Get latent for info
        mask_3ch = (mask_01 * 2 - 1).expand(-1, 3, -1, -1, -1).contiguous()
        with torch.no_grad():
            latent = vae_lora.encode(mask_3ch)
            latent = vae_lora.project_in(latent)
    return recon_mask, latent


def main():
    parser = argparse.ArgumentParser(
        description="VAE-Project Mask Reconstruction Inference"
    )
    parser.add_argument(
        "--mask_video", type=str, required=True, help="Path to input mask video"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to trained MaskVAEAdapter checkpoint (optional if --baseline or --maskvae_path)",
    )
    parser.add_argument(
        "--maskvae_path",
        type=str,
        default=None,
        help="Path to trained MaskVAE stage1 checkpoint (standalone mask VAE without Wan VAE)",
    )
    parser.add_argument(
        "--vae_lora_path",
        type=str,
        default=None,
        help="Path to trained MaskVAELoRA checkpoint (vae-lora mode stage1)",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path to Wan VAE checkpoint (required for adapter/baseline mode)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline mode using raw VAE without adapter",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both adapter and baseline, output comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/reconstruction",
        help="Output directory",
    )
    parser.add_argument(
        "--num_frames", type=int, default=9, help="Number of frames to process"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="384x672",
        help="Target resolution HxW (e.g., 384x672)",
    )
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Compute dtype",
    )

    args = parser.parse_args()

    # Determine mode based on arguments
    if args.vae_lora_path:
        args.mode = "vae_lora"
    elif args.maskvae_path:
        args.mode = "maskvae"
    elif args.baseline and args.compare:
        args.mode = "compare"
    elif args.baseline:
        args.mode = "baseline"
    elif args.adapter_path:
        args.mode = "adapter"
    else:
        parser.error(
            "Must specify one of: --vae_lora_path, --maskvae_path, --adapter_path, or --baseline"
        )

    # Validate vae_path requirement
    if args.mode in ["baseline", "adapter", "compare", "vae_lora"] and args.vae_path is None:
        parser.error("--vae_path is required for adapter/baseline/compare/vae_lora mode")

    if args.mode == "adapter" and args.adapter_path is None:
        parser.error("--adapter_path is required for adapter mode")

    if args.mode == "compare" and args.adapter_path is None:
        parser.error("--adapter_path is required for --compare mode")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    vae = None
    maskvae = None
    adapter = None
    vae_lora = None

    if args.mode == "maskvae":
        print(f"Loading MaskVAE (stage1) from {args.maskvae_path}...")
        maskvae = load_mask_vae(args.maskvae_path, device=device)
        maskvae = maskvae.to(dtype).eval()
    elif args.mode == "vae_lora":
        print(f"Loading MaskVAELoRA from {args.vae_lora_path}...")
        # Load config and state from checkpoint
        state = load_mask_vae_lora_state(args.vae_lora_path, device=device)
        config = MaskVAELoRAConfig(
            lora_rank=state["config"]["lora_rank"],
            lora_alpha=state["config"]["lora_alpha"],
            proj_hidden=state["config"]["proj_hidden"],
            proj_res_blocks=state["config"]["proj_res_blocks"],
            proj_use_attention=state["config"]["proj_use_attention"],
            proj_dropout=state["config"].get("proj_dropout", 0.0),
        )
        vae_lora = MaskVAELoRA(
            vae_pth=args.vae_path,
            config=config,
            dtype=dtype,
            device=device,
        )
        # Load trained weights
        vae_lora.project_in.load_state_dict(state["project_in"])
        vae_lora.load_state_dict(state["decoder_lora"], strict=False)
        vae_lora = vae_lora.eval()
        print(f"  LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
        print(f"  Project hidden: {config.proj_hidden}, res_blocks: {config.proj_res_blocks}")
    else:
        print(f"Loading VAE from {args.vae_path}...")
        vae = WanVAE(vae_pth=args.vae_path, dtype=dtype, device=str(device))

        if args.mode in ["adapter", "compare"]:
            print(f"Loading MaskVAEAdapter from {args.adapter_path}...")
            adapter = load_mask_adapter(args.adapter_path, device=device)
            adapter = adapter.to(dtype).eval()

    print(f"Loading mask video from {args.mask_video}...")
    mask = load_mask_video(args.mask_video, args.num_frames)

    if args.resolution:
        h, w = map(int, args.resolution.split("x"))
        mask = resize_video(mask, (h, w))

    C, T, H, W = mask.shape
    print(f"Mask shape: C={C}, T={T}, H={H}, W={W}")

    mask_tensor = mask.unsqueeze(0).to(device, dtype)  # [1, 1, T, H, W]
    mask_cpu = mask_tensor.squeeze(0).float().cpu()  # [1, T, H, W]

    results = {}

    if args.mode == "maskvae":
        print("\nRunning MaskVAE (stage1) reconstruction...")
        with torch.no_grad():
            recon_maskvae, latent = reconstruct_with_maskvae(
                mask_tensor, maskvae, dtype
            )
        recon_maskvae_cpu = recon_maskvae.squeeze(0).float().cpu()
        metrics_maskvae = compute_metrics(mask_cpu, recon_maskvae_cpu)
        results["maskvae"] = {
            "recon": recon_maskvae_cpu,
            "metrics": metrics_maskvae,
            "latent_shape": latent.shape,
        }
        print(f"MaskVAE Metrics:")
        print(f"  MSE:  {metrics_maskvae['mse']:.6f}")
        print(f"  PSNR: {metrics_maskvae['psnr']:.2f} dB")
        print(f"  MAE:  {metrics_maskvae['mae']:.6f}")
        print(f"  Latent shape: {latent.shape}")

    if args.mode == "vae_lora":
        print("\nRunning MaskVAELoRA (vae-lora stage1) reconstruction...")
        with torch.no_grad():
            recon_vae_lora, latent = reconstruct_with_vae_lora(
                mask_tensor, vae_lora, dtype
            )
        recon_vae_lora_cpu = recon_vae_lora.squeeze(0).float().cpu()
        metrics_vae_lora = compute_metrics(mask_cpu, recon_vae_lora_cpu)
        results["vae_lora"] = {
            "recon": recon_vae_lora_cpu,
            "metrics": metrics_vae_lora,
            "latent_shape": latent.shape,
        }
        print(f"VAE-LoRA Metrics:")
        print(f"  MSE:  {metrics_vae_lora['mse']:.6f}")
        print(f"  PSNR: {metrics_vae_lora['psnr']:.2f} dB")
        print(f"  MAE:  {metrics_vae_lora['mae']:.6f}")
        print(f"  Latent shape: {latent.shape}")

    if args.mode in ["baseline", "compare"]:
        print("\nRunning baseline reconstruction (raw VAE)...")
        with torch.no_grad():
            recon_baseline = reconstruct_with_raw_vae(mask_tensor, vae, dtype)
        recon_baseline_cpu = recon_baseline.squeeze(0).float().cpu()
        metrics_baseline = compute_metrics(mask_cpu, recon_baseline_cpu)
        results["baseline"] = {"recon": recon_baseline_cpu, "metrics": metrics_baseline}
        print(f"Baseline Metrics:")
        print(f"  MSE:  {metrics_baseline['mse']:.6f}")
        print(f"  PSNR: {metrics_baseline['psnr']:.2f} dB")
        print(f"  MAE:  {metrics_baseline['mae']:.6f}")

    if args.mode in ["adapter", "compare"]:
        print("\nRunning adapter reconstruction...")
        with torch.no_grad():
            recon_adapter, latent = adapter(mask_tensor, vae, return_latent=True)
        recon_adapter_cpu = recon_adapter.squeeze(0).float().cpu()
        metrics_adapter = compute_metrics(mask_cpu, recon_adapter_cpu)
        results["adapter"] = {
            "recon": recon_adapter_cpu,
            "metrics": metrics_adapter,
            "latent_shape": latent.shape,
        }
        print(f"Adapter Metrics:")
        print(f"  MSE:  {metrics_adapter['mse']:.6f}")
        print(f"  PSNR: {metrics_adapter['psnr']:.2f} dB")
        print(f"  MAE:  {metrics_adapter['mae']:.6f}")
        print(f"  Latent shape: {latent.shape}")

    base_name = Path(args.mask_video).stem
    original_frames = tensor_to_video_frames(mask_cpu)
    original_rgb = np.repeat(original_frames, 3, axis=-1)
    original_path = os.path.join(args.output_dir, f"{base_name}_original.mp4")

    print(f"\nSaving outputs to {args.output_dir}/")
    save_video(original_rgb, original_path, args.fps)
    saved_files = [f"Original: {original_path}"]

    def save_result_outputs(recon_tensor, metrics_dict, suffix, model_path=None):
        recon_frames_arr = tensor_to_video_frames(recon_tensor)
        recon_rgb_arr = np.repeat(recon_frames_arr, 3, axis=-1)

        recon_out_path = os.path.join(args.output_dir, f"{base_name}_{suffix}.mp4")
        comparison_out_path = os.path.join(
            args.output_dir, f"{base_name}_{suffix}_comparison.mp4"
        )
        grid_out_path = os.path.join(args.output_dir, f"{base_name}_{suffix}_grid.png")
        diff_out_path = os.path.join(args.output_dir, f"{base_name}_{suffix}_diff.mp4")
        metrics_out_path = os.path.join(
            args.output_dir, f"{base_name}_{suffix}_metrics.txt"
        )

        save_video(recon_rgb_arr, recon_out_path, args.fps)
        save_comparison_grid(original_rgb, recon_rgb_arr, comparison_out_path, args.fps)
        save_frame_grid(mask_cpu, recon_tensor, grid_out_path)

        diff_tensor = (mask_cpu - recon_tensor).abs()
        diff_normalized = (
            diff_tensor / diff_tensor.max() if diff_tensor.max() > 0 else diff_tensor
        )
        diff_frames_arr = tensor_to_video_frames(diff_normalized * 2 - 1)
        diff_rgb_arr = np.repeat(diff_frames_arr, 3, axis=-1)
        save_video(diff_rgb_arr, diff_out_path, args.fps)

        with open(metrics_out_path, "w") as mf:
            mf.write(f"Mask Video: {args.mask_video}\n")
            mf.write(f"Mode: {suffix}\n")
            mf.write(f"Model: {model_path}\n")
            mf.write(f"Resolution: {H}x{W}\n")
            mf.write(f"Frames: {T}\n")
            mf.write(f"\nMetrics:\n")
            mf.write(f"  MSE:  {metrics_dict['mse']:.6f}\n")
            mf.write(f"  PSNR: {metrics_dict['psnr']:.2f} dB\n")
            mf.write(f"  MAE:  {metrics_dict['mae']:.6f}\n")

        return [
            recon_out_path,
            comparison_out_path,
            grid_out_path,
            diff_out_path,
            metrics_out_path,
        ]

    if "maskvae" in results:
        out_files = save_result_outputs(
            results["maskvae"]["recon"],
            results["maskvae"]["metrics"],
            "maskvae",
            args.maskvae_path,
        )
        saved_files.extend([f"MaskVAE: {fp}" for fp in out_files])

    if "vae_lora" in results:
        out_files = save_result_outputs(
            results["vae_lora"]["recon"],
            results["vae_lora"]["metrics"],
            "vae_lora",
            args.vae_lora_path,
        )
        saved_files.extend([f"VAE-LoRA: {fp}" for fp in out_files])

    if "baseline" in results:
        out_files = save_result_outputs(
            results["baseline"]["recon"],
            results["baseline"]["metrics"],
            "baseline",
            args.vae_path,
        )
        saved_files.extend([f"Baseline: {fp}" for fp in out_files])

    if "adapter" in results:
        out_files = save_result_outputs(
            results["adapter"]["recon"],
            results["adapter"]["metrics"],
            "adapter",
            args.adapter_path,
        )
        saved_files.extend([f"Adapter: {fp}" for fp in out_files])

    if args.compare and "baseline" in results and "adapter" in results:
        improvement_mse = (
            (
                results["baseline"]["metrics"]["mse"]
                - results["adapter"]["metrics"]["mse"]
            )
            / results["baseline"]["metrics"]["mse"]
            * 100
        )
        improvement_psnr = (
            results["adapter"]["metrics"]["psnr"]
            - results["baseline"]["metrics"]["psnr"]
        )
        improvement_mae = (
            (
                results["baseline"]["metrics"]["mae"]
                - results["adapter"]["metrics"]["mae"]
            )
            / results["baseline"]["metrics"]["mae"]
            * 100
        )
        print("\n=== Comparison ===")
        print(f"  MSE  improvement: {improvement_mse:+.2f}%")
        print(f"  PSNR improvement: {improvement_psnr:+.2f} dB")
        print(f"  MAE  improvement: {improvement_mae:+.2f}%")

        compare_path = os.path.join(args.output_dir, f"{base_name}_compare_metrics.txt")
        with open(compare_path, "w") as f:
            f.write(f"Mask Video: {args.mask_video}\n")
            f.write(f"Adapter: {args.adapter_path}\n")
            f.write(f"Resolution: {H}x{W}\n")
            f.write(f"Frames: {T}\n\n")
            f.write("=== Baseline (Raw VAE) ===\n")
            f.write(f"  MSE:  {results['baseline']['metrics']['mse']:.6f}\n")
            f.write(f"  PSNR: {results['baseline']['metrics']['psnr']:.2f} dB\n")
            f.write(f"  MAE:  {results['baseline']['metrics']['mae']:.6f}\n\n")
            f.write("=== Adapter ===\n")
            f.write(f"  MSE:  {results['adapter']['metrics']['mse']:.6f}\n")
            f.write(f"  PSNR: {results['adapter']['metrics']['psnr']:.2f} dB\n")
            f.write(f"  MAE:  {results['adapter']['metrics']['mae']:.6f}\n\n")
            f.write("=== Improvement ===\n")
            f.write(f"  MSE:  {improvement_mse:+.2f}%\n")
            f.write(f"  PSNR: {improvement_psnr:+.2f} dB\n")
            f.write(f"  MAE:  {improvement_mae:+.2f}%\n")
        saved_files.append(f"Comparison: {compare_path}")

    print("\nOutputs saved:")
    for f in saved_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
