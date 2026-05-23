# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
MaskVAE: Lightweight 3D VAE for single-channel mask encoding/decoding.

This module implements a specialized VAE for alpha masks that outputs latents
compatible with Wan VAE (16 channels, same spatial/temporal stride).

Architecture:
    - Encoder: Conv3d → ResBlocks → Downsample (3 stages) → Bottleneck MLP → Latent
    - Decoder: Latent → Bottleneck MLP → Upsample (3 stages) → ResBlocks → Conv3d

Stride matches Wan VAE:
    - Temporal: 4x (from 2 temporal downsamples)
    - Spatial: 8x (from 3 spatial downsamples)

Two-stage training (mask-vae-project mode):
    Stage 1: Train MaskVAE only (mask reconstruction)
    Stage 2: Freeze MaskVAE, train proj_in/proj_out + LoRA (with mask reconstruction loss)

Single-stage training (mask-vae-joint mode):
    Train MaskVAE + proj_in/proj_out + LoRA end-to-end
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange


class GroupNorm3d(nn.GroupNorm):
    """GroupNorm for 3D video tensors (B, C, T, H, W)."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = input.shape
        x = rearrange(input, "b c t h w -> (b t) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)
        return x


class ResBlock3d(nn.Module):
    """
    Residual block for 3D video with optional GroupNorm.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        use_group_norm: Whether to use GroupNorm (default: True)
        num_groups: Number of groups for GroupNorm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_group_norm: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_group_norm:
            self.residual = nn.Sequential(
                GroupNorm3d(num_groups, in_channels),
                nn.SiLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                GroupNorm3d(num_groups, out_channels),
                nn.SiLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.residual = nn.Sequential(
                nn.SiLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            )

        # Skip connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.residual(x)


class Downsample3d(nn.Module):
    """
    Downsample block for 3D video.

    Args:
        channels: Number of channels
        temporal: Whether to downsample temporally (stride 2 in T)
    """

    def __init__(self, channels: int, temporal: bool = True):
        super().__init__()
        self.temporal = temporal

        if temporal:
            # Stride (2, 2, 2) for temporal + spatial
            self.conv = nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
            )
        else:
            # Stride (1, 2, 2) for spatial only
            self.conv = nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3d(nn.Module):
    """
    Upsample block for 3D video.

    Args:
        channels: Number of channels
        temporal: Whether to upsample temporally (scale 2 in T)
    """

    def __init__(self, channels: int, temporal: bool = True):
        super().__init__()
        self.temporal = temporal
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal:
            # Upsample (2, 2, 2) for temporal + spatial
            x = F.interpolate(
                x.float(),
                scale_factor=(2.0, 2.0, 2.0),
                mode="trilinear",
                align_corners=False,
            ).type_as(x)
        else:
            # Upsample (1, 2, 2) for spatial only
            x = F.interpolate(
                x.float(),
                scale_factor=(1.0, 2.0, 2.0),
                mode="trilinear",
                align_corners=False,
            ).type_as(x)
        return self.conv(x)


class BottleneckMLP(nn.Module):
    """
    MLP block in the bottleneck of the VAE.

    Applies channel-wise MLP with optional residual connection.

    Args:
        channels: Number of input/output channels
        mlp_ratio: Hidden dimension ratio (default: 4)
        depth: Number of MLP layers (default: 1)
        residual: Whether to use residual connection (default: True)
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: int = 4,
        depth: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        hidden = channels * mlp_ratio

        layers = []
        for i in range(depth):
            in_ch = channels if i == 0 else hidden
            out_ch = channels if i == depth - 1 else hidden
            layers.extend(
                [
                    nn.Linear(in_ch, out_ch),
                    nn.SiLU() if i < depth - 1 else nn.Identity(),
                ]
            )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # Reshape to [B*T*H*W, C] for MLP
        x_flat = rearrange(x, "b c t h w -> (b t h w) c")
        out = self.mlp(x_flat)
        out = rearrange(out, "(b t h w) c -> b c t h w", b=B, t=T, h=H, w=W)

        if self.residual:
            return x + out
        return out


class MaskVAEEncoder(nn.Module):
    """
    Encoder for MaskVAE.

    Architecture:
        Input (1ch) → hidden → downsample stages → bottleneck → latent (16ch)

    Args:
        hidden_channels: Base hidden channel dimension (default: 96)
        latent_channels: Latent space channels (default: 16)
        num_res_blocks: Number of residual blocks per stage (default: 2)
        dim_mult: Channel multipliers for each stage (default: [1, 2, 4])
        temporal_downsample: Which stages do temporal downsampling (default: [False, True, True])
        use_group_norm: Whether to use GroupNorm (default: True)
        num_groups: Groups for GroupNorm (default: 8)
        mlp_ratio: MLP hidden ratio in bottleneck (default: 4)
        mlp_depth: Number of MLP layers in bottleneck (default: 1)
    """

    def __init__(
        self,
        hidden_channels: int = 96,
        latent_channels: int = 16,
        num_res_blocks: int = 2,
        dim_mult: Optional[List[int]] = None,
        temporal_downsample: Optional[List[bool]] = None,
        use_group_norm: bool = True,
        num_groups: int = 8,
        mlp_ratio: int = 4,
        mlp_depth: int = 1,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4]
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels

        # Input conv: 1 → hidden
        self.conv_in = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)

        # Downsample stages
        self.down_blocks = nn.ModuleList()
        dims = [hidden_channels * m for m in dim_mult]
        in_ch = hidden_channels

        for i, out_ch in enumerate(dims):
            stage = nn.ModuleList()
            # ResBlocks
            for _ in range(num_res_blocks):
                stage.append(ResBlock3d(in_ch, out_ch, use_group_norm, num_groups))
                in_ch = out_ch
            # Downsample
            stage.append(Downsample3d(out_ch, temporal=temporal_downsample[i]))
            self.down_blocks.append(stage)

        # Bottleneck MLP
        final_ch = dims[-1]
        self.bottleneck_mlp = BottleneckMLP(
            channels=final_ch,
            mlp_ratio=mlp_ratio,
            depth=mlp_depth,
            residual=True,
        )

        # Output conv: final_ch → latent_channels
        self.conv_out = nn.Sequential(
            GroupNorm3d(num_groups, final_ch) if use_group_norm else nn.Identity(),
            nn.SiLU(),
            nn.Conv3d(final_ch, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode mask to latent.

        Args:
            x: [B, 1, T, H, W] mask in range [-1, 1]

        Returns:
            z: [B, 16, T/4, H/8, W/8] latent
        """
        x = self.conv_in(x)

        for stage in self.down_blocks:
            stage_layers = stage.children() if hasattr(stage, "children") else [stage]
            for layer in stage_layers:
                x = layer(x)

        x = self.bottleneck_mlp(x)
        z = self.conv_out(x)

        return z


class MaskVAEDecoder(nn.Module):
    """
    Decoder for MaskVAE.

    Architecture:
        Latent (16ch) → bottleneck → upsample stages → hidden → Output (1ch)

    Args:
        hidden_channels: Base hidden channel dimension (default: 96)
        latent_channels: Latent space channels (default: 16)
        num_res_blocks: Number of residual blocks per stage (default: 2)
        dim_mult: Channel multipliers for each stage (default: [1, 2, 4])
        temporal_upsample: Which stages do temporal upsampling (default: [True, True, False])
        use_group_norm: Whether to use GroupNorm (default: True)
        num_groups: Groups for GroupNorm (default: 8)
        mlp_ratio: MLP hidden ratio in bottleneck (default: 4)
        mlp_depth: Number of MLP layers in bottleneck (default: 1)
    """

    def __init__(
        self,
        hidden_channels: int = 96,
        latent_channels: int = 16,
        num_res_blocks: int = 2,
        dim_mult: Optional[List[int]] = None,
        temporal_upsample: Optional[List[bool]] = None,
        use_group_norm: bool = True,
        num_groups: int = 8,
        mlp_ratio: int = 4,
        mlp_depth: int = 1,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4]
        if temporal_upsample is None:
            temporal_upsample = [True, True, False]  # Reverse of encoder

        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels

        # Dimensions in reverse order
        dims = [hidden_channels * m for m in dim_mult[::-1]]  # [4, 2, 1] * hidden
        final_ch = dims[0]

        # Input conv: latent_channels → final_ch
        self.conv_in = nn.Conv3d(latent_channels, final_ch, kernel_size=3, padding=1)

        # Bottleneck MLP
        self.bottleneck_mlp = BottleneckMLP(
            channels=final_ch,
            mlp_ratio=mlp_ratio,
            depth=mlp_depth,
            residual=True,
        )

        # Upsample stages
        self.up_blocks = nn.ModuleList()
        in_ch = final_ch

        for i, out_ch in enumerate(dims[1:] + [hidden_channels]):
            stage = nn.ModuleList()
            # Upsample first
            stage.append(Upsample3d(in_ch, temporal=temporal_upsample[i]))
            # ResBlocks
            for j in range(num_res_blocks + 1):
                stage.append(
                    ResBlock3d(
                        in_ch if j == 0 else out_ch, out_ch, use_group_norm, num_groups
                    )
                )
            in_ch = out_ch
            self.up_blocks.append(stage)

        # Output conv: hidden → 1
        self.conv_out = nn.Sequential(
            GroupNorm3d(num_groups, hidden_channels)
            if use_group_norm
            else nn.Identity(),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(
        self, z: torch.Tensor, target_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """
        Decode latent to mask.

        Args:
            z: [B, 16, T', H', W'] latent
            target_shape: Optional (T, H, W) to resize output to match original input

        Returns:
            x: [B, 1, T, H, W] reconstructed mask in range [-1, 1]
        """
        x = self.conv_in(z)
        x = self.bottleneck_mlp(x)

        for stage in self.up_blocks:
            stage_layers = stage.children() if hasattr(stage, "children") else [stage]
            for layer in stage_layers:
                x = layer(x)

        x = self.conv_out(x)

        if target_shape is not None:
            T, H, W = target_shape
            x = F.interpolate(
                x.float(), size=(T, H, W), mode="trilinear", align_corners=False
            ).type_as(x)

        return x


class MaskVAE(nn.Module):
    """
    Lightweight 3D VAE for single-channel mask encoding/decoding.

    Outputs latents compatible with Wan VAE:
    - 16 channels
    - Same stride (4x temporal, 8x spatial)

    Args:
        hidden_channels: Base hidden channel dimension (default: 96)
        latent_channels: Latent space channels (default: 16)
        num_res_blocks: Number of residual blocks per stage (default: 2)
        dim_mult: Channel multipliers for each stage (default: [1, 2, 4])
        temporal_downsample: Which stages do temporal downsampling (default: [False, True, True])
        use_group_norm: Whether to use GroupNorm (default: True)
        num_groups: Groups for GroupNorm (default: 8)
        mlp_ratio: MLP hidden ratio in bottleneck (default: 4)
        mlp_depth: Number of MLP layers in bottleneck (default: 1)
    """

    def __init__(
        self,
        hidden_channels: int = 96,
        latent_channels: int = 16,
        num_res_blocks: int = 2,
        dim_mult: Optional[List[int]] = None,
        temporal_downsample: Optional[List[bool]] = None,
        use_group_norm: bool = True,
        num_groups: int = 8,
        mlp_ratio: int = 4,
        mlp_depth: int = 1,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4]
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.dim_mult = dim_mult
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        self.encoder = MaskVAEEncoder(
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            dim_mult=dim_mult,
            temporal_downsample=temporal_downsample,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            mlp_ratio=mlp_ratio,
            mlp_depth=mlp_depth,
        )

        self.decoder = MaskVAEDecoder(
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            dim_mult=dim_mult,
            temporal_upsample=self.temporal_upsample,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            mlp_ratio=mlp_ratio,
            mlp_depth=mlp_depth,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Encode mask to latent.

        Args:
            mask: [B, 1, T, H, W] in range [-1, 1]

        Returns:
            z: [B, 16, T/4, H/8, W/8] latent
        """
        return self.encoder(mask)

    def decode(
        self, z: torch.Tensor, target_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """
        Decode latent to mask.

        Args:
            z: [B, 16, T', H', W'] latent
            target_shape: Optional (T, H, W) to resize output

        Returns:
            mask: [B, 1, T, H, W] in range [-1, 1]
        """
        return self.decoder(z, target_shape=target_shape)

    def forward(
        self,
        mask: torch.Tensor,
        return_latent: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full forward pass: encode → decode.

        Args:
            mask: [B, 1, T, H, W] in range [-1, 1]
            return_latent: If True, also return the latent z

        Returns:
            recon_mask: [B, 1, T, H, W] reconstructed mask
            latent: [B, 16, T', H', W'] (only if return_latent=True)
        """
        input_shape = (mask.size(2), mask.size(3), mask.size(4))
        z = self.encode(mask)
        recon = self.decode(z, target_shape=input_shape)

        if return_latent:
            return recon, z
        return recon, None


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 feature extractor for perceptual loss.

    Extracts features from multiple layers of pretrained VGG16.
    """

    def __init__(self, layers: Optional[List[str]] = None, requires_grad: bool = False):
        super().__init__()
        if layers is None:
            # relu1_2, relu2_2, relu3_3, relu4_3
            layers = ["3", "8", "15", "22"]

        from torchvision import models

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.layers = layers
        self.max_layer = max(int(l) for l in layers) + 1

        # Only keep layers up to max needed
        self.features = nn.Sequential(*list(vgg.features.children())[: self.max_layer])

        # Freeze VGG
        for param in self.features.parameters():
            param.requires_grad = requires_grad

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    mean: torch.Tensor
    std: torch.Tensor

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from VGG16.

        Args:
            x: [B, 3, H, W] in range [-1, 1]

        Returns:
            List of feature maps from specified layers
        """
        # Normalize from [-1, 1] to ImageNet stats
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if str(i) in self.layers:
                features.append(x)
        return features


class MaskVAELoss(nn.Module):
    """
    Loss function for MaskVAE training.

    Combines:
    - SmoothL1/L1 reconstruction loss
    - Spatial gradient loss (edge sharpness)
    - Temporal gradient loss (temporal consistency)
    - Edge-weighted loss (focus on mask boundaries)
    - Perceptual loss (VGG16 feature matching)

    Args:
        rec_loss_type: "smoothl1" or "l1" (default: "smoothl1")
        grad_weight: Weight for spatial gradient loss (default: 0.2)
        temporal_grad_weight: Weight for temporal gradient loss (default: 0.05)
        edge_weight: Weight for edge-weighted loss (default: 0.1)
        edge_scale: Scale factor for edge weighting (default: 2.0)
        perceptual_weight: Weight for perceptual loss (default: 0.0, disabled)
        perceptual_layers: VGG16 layers to use (default: ["3", "8", "15", "22"])
    """

    def __init__(
        self,
        rec_loss_type: str = "smoothl1",
        grad_weight: float = 0.2,
        temporal_grad_weight: float = 0.05,
        edge_weight: float = 0.1,
        edge_scale: float = 2.0,
        perceptual_weight: float = 0.0,
        perceptual_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        self.rec_loss_type = rec_loss_type
        self.grad_weight = grad_weight
        self.temporal_grad_weight = temporal_grad_weight
        self.edge_weight = edge_weight
        self.edge_scale = edge_scale
        self.perceptual_weight = perceptual_weight

        # Sobel kernels
        sobel_h = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
        ).view(1, 1, 1, 3, 3)
        sobel_w = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
        ).view(1, 1, 1, 3, 3)
        self.register_buffer("sobel_h", sobel_h)
        self.register_buffer("sobel_w", sobel_w)

        # VGG16 for perceptual loss
        self.vgg: Optional[VGG16FeatureExtractor] = None
        if perceptual_weight > 0:
            self.vgg = VGG16FeatureExtractor(layers=perceptual_layers)

    sobel_h: torch.Tensor
    sobel_w: torch.Tensor

    def _spatial_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient magnitude."""
        B, C, T, H, W = x.shape
        x_2d = x.reshape(B * T, C, H, W)

        sobel_h = self.sobel_h.squeeze(2).to(x.device, x.dtype)
        sobel_w = self.sobel_w.squeeze(2).to(x.device, x.dtype)

        grad_h = F.conv2d(x_2d, sobel_h, padding=1)
        grad_w = F.conv2d(x_2d, sobel_w, padding=1)

        grad_mag = torch.sqrt(grad_h**2 + grad_w**2 + 1e-8)
        return grad_mag.reshape(B, C, T, H, W)

    def _temporal_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal gradient (frame differences)."""
        return x[:, :, 1:] - x[:, :, :-1]

    def _perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using VGG16 features.

        Args:
            pred: [B, 1, T, H, W] predicted mask
            target: [B, 1, T, H, W] target mask

        Returns:
            Perceptual loss scalar
        """
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)

        B, C, T, H, W = pred.shape

        # Reshape to 2D: [B*T, 1, H, W]
        pred_2d = pred.reshape(B * T, C, H, W)
        target_2d = target.reshape(B * T, C, H, W)

        # Expand single channel to 3 channels
        pred_3ch = pred_2d.expand(-1, 3, -1, -1)
        target_3ch = target_2d.expand(-1, 3, -1, -1)

        # Ensure VGG is on the same device as input
        if next(self.vgg.parameters()).device != pred.device:
            self.vgg = self.vgg.to(pred.device)

        # Extract VGG features
        with torch.amp.autocast('cuda', enabled=False):
            pred_feats = self.vgg(pred_3ch.float())
            with torch.no_grad():
                target_feats = self.vgg(target_3ch.float())

        # Compute L1 loss on each feature layer
        loss = torch.tensor(0.0, device=pred.device)
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf)

        return loss / len(pred_feats)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            pred: Predicted mask [B, 1, T, H, W]
            target: Ground truth mask [B, 1, T, H, W]

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        loss_dict = {}

        # 1. Reconstruction loss
        if self.rec_loss_type == "smoothl1":
            loss_rec = F.smooth_l1_loss(pred, target)
        else:
            loss_rec = F.l1_loss(pred, target)
        loss_dict["loss_rec"] = loss_rec.item()

        # 2. Spatial gradient loss
        grad_pred = self._spatial_gradient(pred)
        with torch.no_grad():
            grad_target = self._spatial_gradient(target)
        loss_grad = F.l1_loss(grad_pred, grad_target)
        loss_dict["loss_grad"] = loss_grad.item()

        # 3. Temporal gradient loss
        if pred.size(2) > 1:
            temp_grad_pred = self._temporal_gradient(pred)
            with torch.no_grad():
                temp_grad_target = self._temporal_gradient(target)
            loss_temp_grad = F.l1_loss(temp_grad_pred, temp_grad_target)
        else:
            loss_temp_grad = torch.tensor(0.0, device=pred.device)
        loss_dict["loss_temp_grad"] = loss_temp_grad.item()

        # 4. Edge-weighted loss (fixed: use per-sample normalization and gradient magnitude)
        with torch.no_grad():
            # Use absolute gradient magnitude for stable weighting
            grad_mag = grad_target.abs()
            # Per-sample normalization (not batch-level) for consistency
            # Shape: [B, 1, T, H, W] -> normalize per sample
            norm = grad_mag.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
            # Clamp weight to avoid extreme values
            edge_weight_map = (1.0 + self.edge_scale * (grad_mag / norm)).clamp(
                1.0, 1.0 + self.edge_scale
            )
        loss_edge = (edge_weight_map * torch.abs(pred - target)).mean()
        loss_dict["loss_edge"] = loss_edge.item()

        # 5. Perceptual loss (VGG16 feature matching)
        if self.perceptual_weight > 0 and self.vgg is not None:
            loss_perceptual = self._perceptual_loss(pred, target)
        else:
            loss_perceptual = torch.tensor(0.0, device=pred.device)
        loss_dict["loss_perceptual"] = loss_perceptual.item()

        # Combine
        total_loss = (
            loss_rec
            + self.grad_weight * loss_grad
            + self.temporal_grad_weight * loss_temp_grad
            + self.edge_weight * loss_edge
            + self.perceptual_weight * loss_perceptual
        )
        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict


def save_mask_vae(mask_vae: MaskVAE, save_path: str):
    """
    Save MaskVAE weights.

    Args:
        mask_vae: MaskVAE instance
        save_path: Path to save (e.g., "checkpoints/mask_vae.pt")
    """
    state_dict = {
        "encoder": mask_vae.encoder.state_dict(),
        "decoder": mask_vae.decoder.state_dict(),
        "config": {
            "hidden_channels": mask_vae.hidden_channels,
            "latent_channels": mask_vae.latent_channels,
            "dim_mult": mask_vae.dim_mult,
            "temporal_downsample": mask_vae.temporal_downsample,
        },
    }
    torch.save(state_dict, save_path)


def load_mask_vae(
    load_path: str,
    device: torch.device = torch.device("cpu"),
    **override_kwargs,
) -> MaskVAE:
    """
    Load MaskVAE from checkpoint.

    Args:
        load_path: Path to checkpoint
        device: Device to load to
        **override_kwargs: Override config values

    Returns:
        MaskVAE instance with loaded weights
    """
    state_dict = torch.load(load_path, map_location=device)

    config = state_dict.get("config", {})
    config.update(override_kwargs)

    mask_vae = MaskVAE(
        hidden_channels=config.get("hidden_channels", 96),
        latent_channels=config.get("latent_channels", 16),
        dim_mult=config.get("dim_mult", [1, 2, 4]),
        temporal_downsample=config.get("temporal_downsample", [False, True, True]),
    )

    mask_vae.encoder.load_state_dict(state_dict["encoder"])
    mask_vae.decoder.load_state_dict(state_dict["decoder"])

    return mask_vae.to(device)
