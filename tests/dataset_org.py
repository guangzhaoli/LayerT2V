# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
Layered Video Dataset for multi-layer video generation training.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import decord

    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("Warning: decord not installed. Using torchvision for video loading.")

try:
    import torchvision.io as tvio

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class LayeredVideoDataset(Dataset):
    """
    Multi-layer video dataset for training layered video generation.

    Dataset structure:
        data_root/
            video_001/
                full_video.mp4   # Complete video (RGB)
                background.mp4   # Background (RGB)
                mask.mp4         # Alpha mask (grayscale, continuous 0-1)
                metadata.json    # Metadata and captions
            video_002/
                ...

    metadata.json format:
        {
            "paths": {
                "background_video": "...",
                "mask_video": "...",
                "full_video": "..."
            },
            "captions": {
                "full": "Complete video description",
                "foreground": "Foreground description",
                "background": "Background description"
            }
        }

    Returns:
        full_video: [3, T, H, W] float, range [-1, 1]
        background: [3, T, H, W] float, range [-1, 1]
        foreground: [3, T, H, W] float, range [-1, 1] (computed as full_video * mask)
        mask: [1, T, H, W] float, range [0, 1]
        caption: str (formatted multi-layer prompt)
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 81,
        resolution: Tuple[int, int] = (480, 720),  # (H, W)
        fps: int = 24,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        no_val_split: bool = False,
        frame_sampling: str = "continuous",
    ):
        """
        Initialize the dataset.

        Args:
            data_root: Root directory containing video samples
            num_frames: Number of frames to sample (should be 4n+1)
            resolution: Target resolution (H, W)
            fps: Target FPS for sampling
            split: "train" or "val"
            split_ratio: Ratio of training samples
            seed: Random seed for train/val split
            no_val_split: If True, use all data for training (no val split)
            frame_sampling: Sampling strategy - "uniform" (evenly spaced) or "continuous" (random start + consecutive)
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.resolution = resolution
        self.fps = fps
        self.split = split
        self.frame_sampling = frame_sampling

        # Find all video samples
        self.samples = self._find_samples()

        # Split into train/val (or use all for train if no_val_split)
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        if no_val_split:
            # Use all data for both train and val (val will be same as train)
            self.indices = indices
        else:
            split_idx = int(len(indices) * split_ratio)
            if split == "train":
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]

        print(f"LayeredVideoDataset [{split}]: {len(self.indices)} samples")

    def _find_samples(self) -> List[Path]:
        """Find all valid video samples in the data root."""
        samples = []
        for sample_dir in self.data_root.iterdir():
            if not sample_dir.is_dir():
                continue

            # Check required files exist
            metadata_path = sample_dir / "metadata.json"
            full_video_path = sample_dir / "full_video.mp4"
            background_path = sample_dir / "background.mp4"
            mask_path = sample_dir / "mask.mp4"

            if all(
                p.exists()
                for p in [metadata_path, full_video_path, background_path, mask_path]
            ):
                samples.append(sample_dir)
            else:
                # Try to find files from metadata
                if metadata_path.exists():
                    samples.append(sample_dir)

        return sorted(samples)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_frame_indices(
        self, total_frames: int, start_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get frame indices based on sampling strategy.

        Args:
            total_frames: Total number of frames in video
            start_ratio: For continuous sampling, the start position as a ratio (0.0-1.0).
                        If None and in train mode, will be random.
                        This allows multiple videos to use the same relative start position.

        Returns:
            Tensor of frame indices to sample
        """
        if total_frames >= self.num_frames:
            if self.frame_sampling == "continuous":
                # Continuous sampling: random start + consecutive frames
                # Training: random start point for data augmentation
                # Validation: fixed start (beginning) for reproducibility
                max_start = total_frames - self.num_frames
                if self.split == "train":
                    if start_ratio is not None:
                        # Use provided ratio to compute start index
                        start_idx = int(start_ratio * max_start)
                    else:
                        start_idx = random.randint(0, max_start)
                else:
                    start_idx = 0
                indices = torch.arange(start_idx, start_idx + self.num_frames)
            else:
                # Uniform sampling: evenly spaced frames across entire video
                indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        else:
            # Video too short: take all frames + repeat last frame
            indices = torch.arange(total_frames)
            pad_indices = torch.full(
                (self.num_frames - total_frames,), total_frames - 1
            )
            indices = torch.cat([indices, pad_indices])

        return indices

    def _load_video(
        self,
        video_path: Union[str, Path],
        is_mask: bool = False,
        start_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Load video from file.

        Args:
            video_path: Path to video file
            is_mask: Whether this is a mask video (grayscale)
            start_ratio: For continuous sampling, the start position as a ratio (0.0-1.0).
                        Pass the same value to multiple videos to keep them synchronized.

        Returns:
            Video tensor [C, T, H, W], range [0, 1]
        """
        video_path = str(video_path)

        if HAS_DECORD:
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)

            # Sample frames based on strategy
            indices = self._get_frame_indices(total_frames, start_ratio)

            frames = vr.get_batch(indices.tolist())  # [T, H, W, C]
            frames = frames.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

        elif HAS_TORCHVISION:
            frames, _, _ = tvio.read_video(video_path, pts_unit="sec")
            frames = frames.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

            total_frames = frames.shape[1]
            indices = self._get_frame_indices(total_frames, start_ratio)
            frames = frames[:, indices]
        else:
            raise RuntimeError(
                "Neither decord nor torchvision available for video loading"
            )

        # Convert mask to grayscale if needed
        if is_mask and frames.shape[0] == 3:
            frames = frames.mean(dim=0, keepdim=True)  # [1, T, H, W]

        # Resize to target resolution
        C, T, H, W = frames.shape
        if (H, W) != self.resolution:
            # Use reshape instead of view (tensor may not be contiguous)
            frames = F.interpolate(
                frames.reshape(C * T, 1, H, W),
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
            ).reshape(C, T, *self.resolution)

        return frames

    def format_layered_prompt(self, captions: Dict[str, str]) -> str:
        """
        Format multi-layer captions into a tagged prompt.

        Args:
            captions: Dictionary with 'full', 'foreground', 'background' keys

        Returns:
            Formatted prompt with <full>, <bg>, <fg>, <mask> tags
        """
        full_caption = captions.get("full", "A video.")
        fg_caption = captions.get("foreground", "An object.")
        bg_caption = captions.get("background", "A background.")

        prompt = (
            f"<full>{full_caption}</full>"
            f"<bg>{bg_caption}</bg>"
            f"<fg>{fg_caption}</fg>"
            f"<mask>foreground alpha mask</mask>"
        )
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - full_video: [3, T, H, W] range [-1, 1]
                - background: [3, T, H, W] range [-1, 1]
                - foreground: [3, T, H, W] range [-1, 1]
                - mask: [1, T, H, W] range [0, 1]
                - caption: str
        """
        sample_idx = self.indices[idx]
        sample_dir = self.samples[sample_idx]

        # Load metadata
        metadata_path = sample_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get file paths - prefer local files, fall back to metadata paths
        paths = metadata.get("paths", {})

        # Default local paths
        local_full = sample_dir / "full_video.mp4"
        local_bg = sample_dir / "background.mp4"
        local_mask = sample_dir / "mask.mp4"

        # Use local paths if they exist, otherwise try metadata paths
        if local_full.exists():
            full_video_path = local_full
        else:
            full_video_path = paths.get("full_video", local_full)

        if local_bg.exists():
            background_path = local_bg
        else:
            background_path = paths.get("background_video", local_bg)

        if local_mask.exists():
            mask_path = local_mask
        else:
            mask_path = paths.get("mask_video", local_mask)

        # Generate shared start_ratio for continuous sampling
        # This ensures all videos (full, background, mask) are sampled from the same relative position
        if self.frame_sampling == "continuous" and self.split == "train":
            start_ratio = random.random()  # Random value in [0.0, 1.0)
        else:
            start_ratio = (
                None  # Will use default behavior (start=0 for val, or uniform sampling)
            )

        # Load videos with shared start_ratio to keep them synchronized
        full_video = self._load_video(
            full_video_path, is_mask=False, start_ratio=start_ratio
        )  # [3, T, H, W]
        background = self._load_video(
            background_path, is_mask=False, start_ratio=start_ratio
        )  # [3, T, H, W]
        mask = self._load_video(
            mask_path, is_mask=True, start_ratio=start_ratio
        )  # [1, T, H, W]

        # Ensure mask is in [0, 1]
        mask = mask.clamp(0, 1)

        # Compute foreground: foreground = full_video * mask
        foreground = full_video * mask  # [3, T, H, W]

        # Normalize RGB videos to [-1, 1]
        full_video = full_video * 2 - 1
        background = background * 2 - 1
        foreground = foreground * 2 - 1

        # Format caption
        captions = metadata.get("captions", {})
        caption = self.format_layered_prompt(captions)

        return {
            "full_video": full_video,  # [3, T, H, W] range [-1, 1]
            "background": background,  # [3, T, H, W] range [-1, 1]
            "foreground": foreground,  # [3, T, H, W] range [-1, 1]
            "mask": mask,  # [1, T, H, W] range [0, 1]
            "caption": caption,  # str
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    return {
        "full_video": torch.stack([b["full_video"] for b in batch]),
        "background": torch.stack([b["background"] for b in batch]),
        "foreground": torch.stack([b["foreground"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }


def collate_fn_mask_only(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Lightweight collate function for MaskVAE training (mask only).

    Only stacks the mask tensor, ignores other video data to save memory.

    Args:
        batch: List of sample dictionaries

    Returns:
        Dictionary with only mask tensor
    """
    return {
        "mask": torch.stack([b["mask"] for b in batch]),
    }


class MaskOnlyDataset(Dataset):
    """
    Lightweight dataset for MaskVAE training - loads only mask videos.

    Saves ~10x memory compared to full LayeredVideoDataset.
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 81,
        resolution: Tuple[int, int] = (480, 720),
        fps: int = 24,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        no_val_split: bool = False,
        frame_sampling: str = "continuous",
    ):
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.resolution = resolution
        self.fps = fps
        self.split = split
        self.frame_sampling = frame_sampling

        self.samples = self._find_samples()

        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        if no_val_split:
            self.indices = indices
        else:
            split_idx = int(len(indices) * split_ratio)
            if split == "train":
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]

        print(f"MaskOnlyDataset [{split}]: {len(self.indices)} samples")

    def _find_samples(self) -> List[Path]:
        samples = []
        for sample_dir in self.data_root.iterdir():
            if not sample_dir.is_dir():
                continue
            mask_path = sample_dir / "mask.mp4"
            if mask_path.exists():
                samples.append(sample_dir)
        return sorted(samples)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_frame_indices(self, total_frames: int) -> torch.Tensor:
        if total_frames >= self.num_frames:
            if self.frame_sampling == "continuous":
                max_start = total_frames - self.num_frames
                start_idx = random.randint(0, max_start) if self.split == "train" else 0
                indices = torch.arange(start_idx, start_idx + self.num_frames)
            else:
                indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        else:
            indices = torch.arange(total_frames)
            pad_indices = torch.full(
                (self.num_frames - total_frames,), total_frames - 1
            )
            indices = torch.cat([indices, pad_indices])
        return indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.indices[idx]
        sample_dir = self.samples[sample_idx]
        mask_path = sample_dir / "mask.mp4"

        if HAS_DECORD:
            vr = decord.VideoReader(str(mask_path))
            total_frames = len(vr)
            indices = self._get_frame_indices(total_frames)
            frames = vr.get_batch(indices.tolist())
            frames = frames.permute(3, 0, 1, 2).float() / 255.0
        elif HAS_TORCHVISION:
            frames, _, _ = tvio.read_video(str(mask_path), pts_unit="sec")
            frames = frames.permute(3, 0, 1, 2).float() / 255.0
            total_frames = frames.shape[1]
            indices = self._get_frame_indices(total_frames)
            frames = frames[:, indices]
        else:
            raise RuntimeError("Neither decord nor torchvision available")

        if frames.shape[0] == 4:
            frames = frames[3:4]
        elif frames.shape[0] == 3:
            frames = frames.mean(dim=0, keepdim=True)

        C, T, H, W = frames.shape
        if (H, W) != self.resolution:
            frames = F.interpolate(
                frames.reshape(C * T, 1, H, W),
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
            ).reshape(C, T, *self.resolution)

        return {"mask": frames.clamp(0, 1)}


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    args = parser.parse_args()

    dataset = LayeredVideoDataset(
        data_root=args.data_root,
        num_frames=args.num_frames,
        split="train",
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"full_video shape: {sample['full_video'].shape}")
        print(f"background shape: {sample['background'].shape}")
        print(f"foreground shape: {sample['foreground'].shape}")
        print(f"mask shape: {sample['mask'].shape}")
        print(f"caption: {sample['caption'][:100]}...")