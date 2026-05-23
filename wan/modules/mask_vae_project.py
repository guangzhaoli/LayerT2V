# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
Mask Latent Projection layers for vae-project, mask-vae-project, and mask-vae-joint modes.

These layers bridge between mask latent space and DiT model latent space:
- project_in: Applied to noisy mask latent before feeding to DiT
- project_out: Applied to predicted mask velocity after DiT output

Architecture: MLP → RMSNorm → learnable gate (init 0) → residual
This ensures minimal distribution change at initialization.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float for numerical stability (like WanRMSNorm)
        input_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(input_dtype)


class GatedMLP(nn.Module):
    """
    MLP with optional RMSNorm and learnable gate for residual connection.

    Structure: y = x + gate * norm(MLP(x))
    - gate is initialized to 0 for stable training start
    - norm is applied after MLP (post-norm)

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        hidden_channels: Hidden layer dimension
        depth: Number of linear layers (default 2)
        norm_type: Normalization type - "none" or "rmsnorm"
        use_residual: Whether to use residual connection
        gate_init: Initial value for gate (0 = identity at start)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        depth: int = 2,
        norm_type: Literal["none", "rmsnorm"] = "none",
        use_residual: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.norm_type = norm_type
        self.use_residual = use_residual and (in_channels == out_channels)
        self.gate_init = gate_init

        # Build MLP layers
        layers = []
        curr_ch = in_channels
        for i in range(depth):
            next_ch = out_channels if i == depth - 1 else hidden_channels
            layers.append(nn.Linear(curr_ch, next_ch))
            if i < depth - 1:
                layers.append(nn.SiLU())
            curr_ch = next_ch
        self.mlp = nn.Sequential(*layers)

        # Post-MLP normalization
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(out_channels)
        else:
            self.norm = nn.Identity()

        # Learnable gate for residual (initialized to gate_init, typically 0)
        if self.use_residual:
            self.gate = nn.Parameter(torch.tensor(gate_init))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        out = self.norm(out)
        if self.use_residual:
            gate = torch.tanh(self.gate)
            return x + gate * out
        return out


class MaskLatentProjectIn(nn.Module):
    """
    Project layer for encoding path (applied to noisy mask latent before DiT).

    Maps 16ch latent → 16ch latent with optional normalization and gated residual.

    Includes learnable pre_scale and pre_bias to align MaskVAE latent distribution
    with Wan VAE latent distribution before the MLP transformation.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_channels: int = 128,
        mlp_depth: int = 2,
        norm_type: Literal["none", "rmsnorm"] = "none",
        use_residual: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.mlp_depth = mlp_depth
        self.norm_type = norm_type
        self.gate_init = gate_init

        # Learnable affine transform to align MaskVAE latent distribution with Wan VAE
        # MaskVAE latents are not standardized, while Wan VAE latents are normalized
        # per-channel. This allows learning the alignment during training.
        self.pre_scale = nn.Parameter(torch.ones(1, latent_channels, 1, 1, 1))
        self.pre_bias = nn.Parameter(torch.zeros(1, latent_channels, 1, 1, 1))

        self.proj = GatedMLP(
            in_channels=latent_channels,
            out_channels=latent_channels,
            hidden_channels=hidden_channels,
            depth=mlp_depth,
            norm_type=norm_type,
            use_residual=use_residual,
            gate_init=gate_init,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply projection to mask latent.

        Args:
            z: [B, 16, T', H', W'] or [16, T', H', W'] mask latent

        Returns:
            z_proj: same shape, projected latent
        """
        if z.dim() == 4:
            # Unbatched: [C, T, H, W] - add batch dim for pre_scale/pre_bias
            z = z.unsqueeze(0)
            z = z * self.pre_scale + self.pre_bias
            z = z.squeeze(0)
            C, T, H, W = z.shape
            z_flat = z.permute(1, 2, 3, 0).reshape(-1, C)
            z_proj = self.proj(z_flat)
            return z_proj.reshape(T, H, W, C).permute(3, 0, 1, 2)
        else:
            # Batched: [B, C, T, H, W]
            # Apply learnable affine to align MaskVAE latents with Wan VAE distribution
            z = z * self.pre_scale + self.pre_bias
            B, C, T, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)
            z_proj = self.proj(z_flat)
            return z_proj.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


class MaskLatentProjectOut(nn.Module):
    """
    Project layer for decoding path (applied to predicted mask velocity after DiT).

    Maps 16ch velocity → 16ch velocity with optional normalization and gated residual.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_channels: int = 128,
        mlp_depth: int = 2,
        norm_type: Literal["none", "rmsnorm"] = "none",
        use_residual: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.mlp_depth = mlp_depth
        self.norm_type = norm_type
        self.gate_init = gate_init

        self.proj = GatedMLP(
            in_channels=latent_channels,
            out_channels=latent_channels,
            hidden_channels=hidden_channels,
            depth=mlp_depth,
            norm_type=norm_type,
            use_residual=use_residual,
            gate_init=gate_init,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply projection to predicted mask velocity.

        Args:
            z: [B, 16, T', H', W'] or [16, T', H', W'] predicted velocity

        Returns:
            z_proj: same shape, projected velocity
        """
        if z.dim() == 4:
            # Unbatched: [C, T, H, W]
            C, T, H, W = z.shape
            z_flat = z.permute(1, 2, 3, 0).reshape(-1, C)
            z_proj = self.proj(z_flat)
            return z_proj.reshape(T, H, W, C).permute(3, 0, 1, 2)
        else:
            # Batched: [B, C, T, H, W]
            B, C, T, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)
            z_proj = self.proj(z_flat)
            return z_proj.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


# ============================================================================
# Legacy aliases for backward compatibility with mask-vae-project mode
# ============================================================================
MaskVAEProjectIn = MaskLatentProjectIn
MaskVAEProjectOut = MaskLatentProjectOut


def save_mask_latent_projects(
    proj_in: MaskLatentProjectIn,
    proj_out: MaskLatentProjectOut,
    save_path: str,
):
    """Save projection layer weights."""
    state_dict = {
        "proj_in": proj_in.state_dict(),
        "proj_out": proj_out.state_dict(),
        "config": {
            "latent_channels": proj_in.latent_channels,
            "hidden_channels": proj_in.hidden_channels,
            "mlp_depth": proj_in.mlp_depth,
            "norm_type": proj_in.norm_type,
            "gate_init": proj_in.gate_init,
        },
    }
    torch.save(state_dict, save_path)


def load_mask_latent_projects(
    load_path: str,
    device: torch.device = torch.device("cpu"),
    hidden_channels: int = 128,
    mlp_depth: int = 2,
    norm_type: Literal["none", "rmsnorm"] = "none",
    use_residual: bool = True,
    gate_init: float = 0.0,
):
    """
    Load projection layers from checkpoint.

    Supports backward compatibility with old checkpoints that lack new config fields.
    """
    state_dict = torch.load(load_path, map_location=device)
    config = state_dict.get("config", {})

    # Load config with fallbacks for backward compatibility
    latent_channels = config.get("latent_channels", 16)
    hidden_channels = config.get("hidden_channels", hidden_channels)
    mlp_depth = config.get("mlp_depth", mlp_depth)
    norm_type_loaded: str = config.get("norm_type", norm_type)
    norm_type_cast: Literal["none", "rmsnorm"] = (
        "rmsnorm" if norm_type_loaded == "rmsnorm" else "none"
    )
    gate_init = config.get("gate_init", gate_init)

    proj_in = MaskLatentProjectIn(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        mlp_depth=mlp_depth,
        norm_type=norm_type_cast,
        use_residual=use_residual,
        gate_init=gate_init,
    )
    proj_out = MaskLatentProjectOut(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        mlp_depth=mlp_depth,
        norm_type=norm_type_cast,
        use_residual=use_residual,
        gate_init=gate_init,
    )

    # Load state dict (handles missing keys gracefully for old checkpoints)
    proj_in.load_state_dict(state_dict["proj_in"], strict=False)
    proj_out.load_state_dict(state_dict["proj_out"], strict=False)

    return proj_in.to(device), proj_out.to(device)


# Legacy aliases for backward compatibility
save_mask_vae_projects = save_mask_latent_projects
load_mask_vae_projects = load_mask_latent_projects
