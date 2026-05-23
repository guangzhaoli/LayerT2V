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
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        jsonl_path: Optional[str] = None,  # ===== 新增 =====
        num_frames: int = 81,
        resolution: Tuple[int, int] = (480, 720),
        fps: int = 24,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        no_val_split: bool = False,
        frame_sampling: str = "continuous",
    ):
        if data_root is None and jsonl_path is None:
            raise ValueError("data_root and jsonl_path cannot both be None")

        self.data_root = Path(data_root) if data_root is not None else None
        self.jsonl_path = jsonl_path
        self.use_jsonl = jsonl_path is not None  # ===== 新增 =====

        self.num_frames = num_frames
        self.resolution = resolution
        self.fps = fps
        self.split = split
        self.frame_sampling = frame_sampling

        # ===== 修改：根据来源加载 samples =====
        if self.use_jsonl:
            self.samples = self._load_jsonl_samples()
        else:
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

        print(
            f"LayeredVideoDataset [{split}] ({'jsonl' if self.use_jsonl else 'dir'}): "
            f"{len(self.indices)} samples"
        )

    # ================= 原逻辑保持不变 =================
    def _find_samples(self) -> List[Path]:
        samples = []
        for sample_dir in self.data_root.iterdir():
            if not sample_dir.is_dir():
                continue

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
                if metadata_path.exists():
                    samples.append(sample_dir)

        return sorted(samples)

    # ================= 新增：jsonl 加载 =================
    def _load_jsonl_samples(self) -> List[Dict]:
        samples = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.indices)

    def _get_frame_indices(
        self, total_frames: int, start_ratio: Optional[float] = None
    ) -> torch.Tensor:
        if total_frames >= self.num_frames:
            if self.frame_sampling == "continuous":
                max_start = total_frames - self.num_frames
                if self.split == "train":
                    if start_ratio is not None:
                        start_idx = int(start_ratio * max_start)
                    else:
                        start_idx = random.randint(0, max_start)
                else:
                    start_idx = 0
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

    def _load_video(
        self,
        video_path: Union[str, Path],
        is_mask: bool = False,
        start_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        video_path = str(video_path)

        if HAS_DECORD:
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            indices = self._get_frame_indices(total_frames, start_ratio)
            frames = vr.get_batch(indices.tolist())
            frames = frames.permute(3, 0, 1, 2).float() / 255.0
        elif HAS_TORCHVISION:
            frames, _, _ = tvio.read_video(video_path, pts_unit="sec")
            frames = frames.permute(3, 0, 1, 2).float() / 255.0
            total_frames = frames.shape[1]
            indices = self._get_frame_indices(total_frames, start_ratio)
            frames = frames[:, indices]
        else:
            raise RuntimeError("Neither decord nor torchvision available")

        if is_mask and frames.shape[0] == 3:
            frames = frames.mean(dim=0, keepdim=True)

        C, T, H, W = frames.shape
        if (H, W) != self.resolution:
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
        sample_idx = self.indices[idx]

        # ================= jsonl 分支 =================
        if self.use_jsonl:
            sample = self.samples[sample_idx]

            full_video_path = sample["vid"]
            mask_path = sample["entry"]["pha_video_path"]
            background_path = sample["entry"]["bg_path"]

            caption_full = sample.get("overall_description", "A video.")
            caption_fg = sample.get("foreground_prompt", "An object.")
            caption_bg = sample.get("background_prompt", "A background.")

        # ================= 原 data_root 分支 =================
        else:
            sample_dir = self.samples[sample_idx]
            metadata_path = sample_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            paths = metadata.get("paths", {})
            local_full = sample_dir / "full_video.mp4"
            local_bg = sample_dir / "background.mp4"
            local_mask = sample_dir / "mask.mp4"

            full_video_path = local_full if local_full.exists() else paths.get(
                "full_video", local_full
            )
            background_path = local_bg if local_bg.exists() else paths.get(
                "background_video", local_bg
            )
            mask_path = local_mask if local_mask.exists() else paths.get(
                "mask_video", local_mask
            )

            captions = metadata.get("captions", {})
            caption_full = captions.get("full", "A video.")
            caption_fg = captions.get("foreground", "An object.")
            caption_bg = captions.get("background", "A background.")

        if self.frame_sampling == "continuous" and self.split == "train":
            # start_ratio = random.random()
            start_ratio = 0.0
        else:
            start_ratio = None

        full_video = self._load_video(full_video_path, False, start_ratio)
        background = self._load_video(background_path, False, start_ratio)
        mask = self._load_video(mask_path, True, start_ratio)

        mask = mask.clamp(0, 1)
        foreground = full_video * mask

        full_video = full_video * 2 - 1
        background = background * 2 - 1
        foreground = foreground * 2 - 1

        return {
            "full_video": full_video,
            "background": background,
            "foreground": foreground,
            "mask": mask,
            "caption_full": caption_full,
            "caption_fg": caption_fg,
            "caption_bg": caption_bg,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Custom collate function for DataLoader."""
    tensor_keys = ["full_video", "background", "foreground", "mask"]
    caption_keys = ["caption_full", "caption_fg", "caption_bg"]
    result = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    result.update({k: [b[k] for b in batch] for k in caption_keys})
    return result


def collate_fn_mask_only(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Lightweight collate for MaskVAE training (mask only)."""
    return {"mask": torch.stack([b["mask"] for b in batch])}




class MaskOnlyDataset(Dataset):
    """
    Lightweight dataset for MaskVAE training - loads only mask videos.

    Saves ~10x memory compared to full LayeredVideoDataset.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        jsonl_path: Optional[str] = None,   # ===== 新增 =====
        num_frames: int = 81,
        resolution: Tuple[int, int] = (480, 720),
        fps: int = 24,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        no_val_split: bool = False,
        frame_sampling: str = "continuous",
    ):
        if data_root is None and jsonl_path is None:
            raise ValueError("data_root and jsonl_path cannot both be None")

        self.data_root = Path(data_root) if data_root is not None else None
        self.jsonl_path = jsonl_path
        self.use_jsonl = jsonl_path is not None  # ===== 新增 =====

        self.num_frames = num_frames
        self.resolution = resolution
        self.fps = fps
        self.split = split
        self.frame_sampling = frame_sampling

        # ===== 修改：根据来源加载 samples =====
        if self.use_jsonl:
            self.samples = self._load_jsonl_samples()
        else:
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

        print(
            f"MaskOnlyDataset [{split}] ({'jsonl' if self.use_jsonl else 'dir'}): "
            f"{len(self.indices)} samples"
        )

    # ================= 原逻辑 =================
    def _find_samples(self) -> List[Path]:
        samples = []
        for sample_dir in self.data_root.iterdir():
            if not sample_dir.is_dir():
                continue
            mask_path = sample_dir / "mask.mp4"
            if mask_path.exists():
                samples.append(sample_dir)
        return sorted(samples)

    # ================= 新增 =================
    def _load_jsonl_samples(self) -> List[Dict]:
        samples = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

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

        # ===== jsonl 分支 =====
        if self.use_jsonl:
            sample = self.samples[sample_idx]
            mask_path = sample["entry"]["pha_video_path"]

        # ===== 原 data_root 分支 =====
        else:
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
    parser.add_argument("--data_root", type=str, required=False)
    parser.add_argument("--jsonl_path", type=str, required=False)
    parser.add_argument("--num_frames", type=int, default=81)
    args = parser.parse_args()

    dataset = LayeredVideoDataset(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
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
        print(f"caption_full: {sample['caption_full'][:100]}...")
        print(f"caption_fg: {sample['caption_fg'][:100]}...")
        print(f"caption_bg: {sample['caption_bg'][:100]}...")