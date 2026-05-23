# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
Mask VAE LoRA utilities for the vae-lora mask mode.

Stage 1:
  - Freeze Wan VAE encoder
  - Project latent with a causal 3D merge block
  - Fine-tune Wan VAE decoder with Conv3d LoRA
Stage 2:
  - Freeze Stage 1 components
  - Train DiT LoRA using projected mask latents
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .vae import CausalConv3d, ResidualBlock, AttentionBlock, _video_vae


_WAN_VAE_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]

_WAN_VAE_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


def _get_causal_padding(conv: CausalConv3d) -> Tuple[int, int, int]:
    if hasattr(conv, "_padding"):
        pad_w = int(conv._padding[0])
        pad_h = int(conv._padding[2])
        pad_t = int(conv._padding[4] // 2)
        return pad_t, pad_h, pad_w
    return conv.padding


class LoRACausalConv3d(CausalConv3d):
    """Causal 3D convolution with LoRA adapters."""

    def __init__(self, base_conv: CausalConv3d, rank: int, alpha: float):
        pad_t, pad_h, pad_w = _get_causal_padding(base_conv)
        super().__init__(
            base_conv.in_channels,
            base_conv.out_channels,
            base_conv.kernel_size,
            base_conv.stride,
            padding=(pad_t, pad_h, pad_w),
            dilation=base_conv.dilation,
            groups=base_conv.groups,
            bias=base_conv.bias is not None,
        )

        with torch.no_grad():
            self.weight.copy_(base_conv.weight)
            if base_conv.bias is not None and self.bias is not None:
                self.bias.copy_(base_conv.bias)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_down = CausalConv3d(
            base_conv.in_channels,
            rank,
            base_conv.kernel_size,
            base_conv.stride,
            padding=(pad_t, pad_h, pad_w),
            dilation=base_conv.dilation,
            groups=base_conv.groups,
            bias=False,
        )
        self.lora_up = CausalConv3d(
            rank,
            base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_normal_(self.lora_down.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / float(rank)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None):
        base = super().forward(x, cache_x)
        if cache_x is None:
            lora = self.lora_down(x)
        else:
            lora = self.lora_down(x, cache_x)
        lora = self.lora_up(lora)
        return base + self.scale * lora


def apply_decoder_lora(module: nn.Module, rank: int, alpha: float) -> None:
    """Recursively replace CausalConv3d layers with LoRA-wrapped versions."""
    for name, child in module.named_children():
        if isinstance(child, CausalConv3d):
            setattr(module, name, LoRACausalConv3d(child, rank, alpha))
        else:
            apply_decoder_lora(child, rank, alpha)


def extract_lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters for saving."""
    state = module.state_dict()
    return {k: v for k, v in state.items() if ".lora_down." in k or ".lora_up." in k}


@dataclass
class MaskVAELoRAConfig:
    lora_rank: int
    lora_alpha: float
    proj_hidden: int = 64
    proj_res_blocks: int = 2
    proj_use_attention: bool = True
    proj_dropout: float = 0.0


class MaskLatentProjectIn3D(nn.Module):
    """Causal 3D latent project-in block (Wan-Alpha style merge block)."""

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        res_blocks: int = 2,
        use_attention: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.res_blocks = res_blocks
        self.use_attention = use_attention
        self.dropout = dropout

        self.in_conv = CausalConv3d(in_channels, hidden_channels, 3, padding=1)
        blocks = []
        for _ in range(res_blocks):
            blocks.append(ResidualBlock(hidden_channels, hidden_channels, dropout))
            if use_attention:
                blocks.append(AttentionBlock(hidden_channels))
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = CausalConv3d(hidden_channels, in_channels, 3, padding=1)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h)
        h = self.out_conv(h)
        return x + torch.tanh(self.gate) * h


class MaskVAELoRA(nn.Module):
    """Stage 1 trainer: frozen encoder + project-in + decoder LoRA."""

    def __init__(
        self,
        vae_pth: str,
        config: MaskVAELoRAConfig,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.config = config

        mean = torch.tensor(_WAN_VAE_MEAN, dtype=dtype, device=device)
        std = torch.tensor(_WAN_VAE_STD, dtype=dtype, device=device)
        self.scale = [mean, 1.0 / std]

        self.model = _video_vae(pretrained_path=vae_pth, z_dim=16)
        self.model.eval().requires_grad_(False)

        # LoRA on decoder path only.
        self.model.conv2 = LoRACausalConv3d(self.model.conv2, config.lora_rank, config.lora_alpha)
        apply_decoder_lora(self.model.decoder, config.lora_rank, config.lora_alpha)

        self.project_in = MaskLatentProjectIn3D(
            hidden_channels=config.proj_hidden,
            res_blocks=config.proj_res_blocks,
            use_attention=config.proj_use_attention,
            dropout=config.proj_dropout,
        )
        self.to(device)

    def encode(self, mask_3ch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode(mask_3ch, self.scale)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents, self.scale)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # Use expand + contiguous for memory efficiency and DDP compatibility
        mask_normalized = mask * 2 - 1
        mask_3ch = mask_normalized.expand(-1, 3, -1, -1, -1).contiguous()
        z = self.encode(mask_3ch)
        z_proj = self.project_in(z)
        recon_3ch = self.decode(z_proj)
        return recon_3ch.mean(dim=1, keepdim=True)


class MaskVAELoRADecoder(nn.Module):
    """Decoder-only wrapper for inference (decoder LoRA applied)."""

    def __init__(
        self,
        vae_pth: str,
        lora_rank: int,
        lora_alpha: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        mean = torch.tensor(_WAN_VAE_MEAN, dtype=dtype, device=device)
        std = torch.tensor(_WAN_VAE_STD, dtype=dtype, device=device)
        self.scale = [mean, 1.0 / std]

        self.model = _video_vae(pretrained_path=vae_pth, z_dim=16)
        self.model.eval().requires_grad_(False)

        self.model.conv2 = LoRACausalConv3d(self.model.conv2, lora_rank, lora_alpha)
        apply_decoder_lora(self.model.decoder, lora_rank, lora_alpha)
        self.to(device)

    def decode(self, zs):
        return [
            self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]


def save_mask_vae_lora(
    save_path: str,
    project_in: MaskLatentProjectIn3D,
    decoder_lora_state: Dict[str, torch.Tensor],
    config: MaskVAELoRAConfig,
) -> None:
    torch.save(
        {
            "project_in": project_in.state_dict(),
            "decoder_lora": decoder_lora_state,
            "config": {
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "proj_hidden": config.proj_hidden,
                "proj_res_blocks": config.proj_res_blocks,
                "proj_use_attention": config.proj_use_attention,
                "proj_dropout": config.proj_dropout,
            },
        },
        save_path,
    )


def load_mask_vae_lora_state(load_path: str, device: torch.device) -> Dict[str, object]:
    return torch.load(load_path, map_location=device)


def build_project_in_from_state(
    state: Dict[str, object], device: torch.device
) -> MaskLatentProjectIn3D:
    cfg = state.get("config", {})
    project_in = MaskLatentProjectIn3D(
        hidden_channels=cfg.get("proj_hidden", 64),
        res_blocks=cfg.get("proj_res_blocks", 2),
        use_attention=cfg.get("proj_use_attention", True),
        dropout=cfg.get("proj_dropout", 0.0),
    ).to(device)
    project_in.load_state_dict(state["project_in"])
    project_in.eval()
    for p in project_in.parameters():
        p.requires_grad = False
    return project_in


def build_decoder_from_state(
    state: Dict[str, object],
    vae_pth: str,
    device: torch.device,
    dtype: torch.dtype,
) -> MaskVAELoRADecoder:
    cfg = state.get("config", {})
    decoder = MaskVAELoRADecoder(
        vae_pth=vae_pth,
        lora_rank=int(cfg.get("lora_rank", 4)),
        lora_alpha=float(cfg.get("lora_alpha", 1.0)),
        dtype=dtype,
        device=device,
    )
    decoder.load_state_dict(state["decoder_lora"], strict=False)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder
