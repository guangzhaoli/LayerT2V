# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
Layered WanModel for multi-layer video generation.

This module extends WanModel with:
1. LayerAdaLN - AdaLN-style layer modulation for each layer category
2. Layer splitting utilities for output processing
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.configuration_utils import register_to_config

from .model import WanModel, sinusoidal_embedding_1d, rope_params, LayeredCrossAttention


class LayerAdaLN(nn.Module):
    """
    Layer-conditioned AdaLN modulation for multi-layer video generation.

    Each layer category gets 6 modulation parameters (shift_attn, scale_attn, gate_attn,
    shift_ffn, scale_ffn, gate_ffn) that are added to the timestep modulation.
    """

    def __init__(self, dim: int, num_layers: int = 4):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        # Zero init: starts as identity, doesn't affect original model behavior
        self.layer_modulation = nn.Parameter(torch.zeros(num_layers, 6, dim))

    def get_modulation(
        self, seq_len: int, tokens_per_layer: int, device: torch.device
    ) -> torch.Tensor:
        """Get per-token layer modulation vectors [L, 6, dim]."""
        if tokens_per_layer <= 0:
            return None
        layer_ids = (torch.arange(seq_len, device=device) // tokens_per_layer).clamp(
            max=self.num_layers - 1
        )
        return self.layer_modulation[layer_ids]


class MaskEncoder(nn.Module):
    """
    Learnable encoder for mask: downsample + channel expansion.

    Combines spatial downsampling (to match VAE latent) with channel expansion
    (1 -> 16 channels) in a single unified module.

    Architecture (matches VAE's 8x spatial, 4x temporal downsampling):
        Stage 1: 1 -> 32ch, stride=(2,2,2) → T/2, H/2, W/2
        Stage 2: 32 -> 32ch, stride=(2,2,2) → T/4, H/4, W/4
        Stage 3: 32 -> 16ch, stride=(1,2,2) → T/4, H/8, W/8

    Args:
        in_channels: Input channels (default: 1)
        out_channels: Output channels (default: 16, matching VAE latent)
        hidden_channels: Hidden layer channels (default: 32)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 16,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 3 stages: downsample + channel expansion
        self.encoder = nn.Sequential(
            # Stage 1: T/2, H/2, W/2, 1 -> hidden
            nn.Conv3d(
                in_channels, hidden_channels, kernel_size=3, stride=(2, 2, 2), padding=1
            ),
            nn.SiLU(),
            # Stage 2: T/4, H/4, W/4, hidden -> hidden
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.SiLU(),
            # Stage 3: T/4, H/8, W/8, hidden -> out (no temporal downsample)
            nn.Conv3d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.encoder:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Encode mask: downsample + channel expansion.

        Args:
            x: Mask tensor [B, 1, T, H, W] in range [-1, 1]
            target_size: Optional target size (T', H', W') to ensure exact match
                        with VAE latent dimensions.

        Returns:
            Encoded mask [B, 16, T', H', W']
        """
        out = self.encoder(x)

        # Ensure exact size match with VAE latent if target specified
        if target_size is not None and out.shape[2:] != torch.Size(target_size):
            out = F.interpolate(
                out, size=target_size, mode="trilinear", align_corners=False
            )

        return out


class MaskDecoder(nn.Module):
    """
    Learnable decoder for mask: upsample + channel contraction.

    Combines spatial upsampling with channel contraction (16 -> 1 channel)
    in a single unified module. Uses transposed convolutions for upsampling.

    Architecture (inverse of MaskEncoder):
        Stage 1: 16 -> 32ch, upsample 2x spatial
        Stage 2: 32 -> 32ch, upsample 2x spatial + 2x temporal
        Stage 3: 32 -> 1ch, upsample 2x spatial + 2x temporal

    Args:
        in_channels: Input channels (default: 16, matching VAE latent)
        out_channels: Output channels (default: 1)
        hidden_channels: Hidden layer channels (default: 32)
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 1,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 3 stages: upsample + channel contraction using ConvTranspose3d
        self.decoder = nn.Sequential(
            # Stage 1: 2x spatial upsample (H*2, W*2), 16 -> hidden
            nn.ConvTranspose3d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                output_padding=(0, 1, 1),
            ),
            nn.SiLU(),
            # Stage 2: 2x temporal + 2x spatial upsample, hidden -> hidden
            nn.ConvTranspose3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=(1, 1, 1),
            ),
            nn.SiLU(),
            # Stage 3: 2x temporal + 2x spatial upsample, hidden -> out
            nn.ConvTranspose3d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=(1, 1, 1),
            ),
        )

        # Refinement layer for sharper edges (residual)
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.decoder:
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.refine:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init last refine layer for residual learning
        last_conv = self.refine[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Decode mask: upsample + channel contraction.

        Args:
            x: Encoded mask tensor [B, 16, T', H', W'] in range [-1, 1]
            target_size: Target size (T, H, W) for final output.

        Returns:
            Decoded mask [B, 1, T, H, W]
        """
        out = self.decoder(x)

        # Ensure exact size match with target
        if target_size is not None and out.shape[2:] != torch.Size(target_size):
            out = F.interpolate(
                out, size=target_size, mode="trilinear", align_corners=False
            )

        # Apply refinement with residual connection
        out = out + self.refine(out)

        return out


class LayeredWanModel(WanModel):
    """
    Extended WanModel for multi-layer video generation.

    Uses LayerAdaLN to distinguish between different layer categories
    (full_video, background, foreground, mask) when concatenated along time dimension.

    The model processes input where layers are concatenated along the time dimension:
        [full_video_latent | background_latent | foreground_latent | mask_latent]
        Each has shape [C, T', H', W'], concatenated to [C, 4*T', H', W']
    """

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        num_output_layers: int = 4,  # Number of layer categories
        mask_mode: str = "vae",  # "vae", "downsample", "downsample-project", "vae-project", "vae-lora", "mask-vae-project", or "mask-vae-joint"
        use_4d_rope: bool = True,  # Use 4D RoPE (L, T, H, W) instead of 3D
        rope_dim_ratios: tuple = None,  # (L, T, H, W) real dimensions, None for default
    ):
        """
        Initialize LayeredWanModel.

        Args:
            num_output_layers: Number of layer categories (4 for full/bg/fg/mask)
            mask_mode: Mask processing mode - "vae", "downsample", "downsample-project", "vae-project", "vae-lora", "mask-vae-project", or "mask-vae-joint"
            use_4d_rope: Whether to use 4D RoPE for (L, T, H, W) position encoding
            rope_dim_ratios: Custom dimension allocation (L, T, H, W) in real dims, must sum to head_dim
            Other args: Same as WanModel
        """
        # Initialize parent class
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )

        self.num_output_layers = num_output_layers
        self.mask_mode = mask_mode

        # Add LayerAdaLN for layer-conditioned modulation (replaces LayerPositionOffset)
        self.layer_adaln = LayerAdaLN(dim, num_output_layers)

        # Add mask encoder/decoder for downsample-project mode
        if mask_mode == "downsample-project":
            self.mask_encoder = MaskEncoder(
                in_channels=1, out_channels=out_dim, hidden_channels=32
            )
            self.mask_decoder = MaskDecoder(
                in_channels=out_dim, out_channels=1, hidden_channels=32
            )
        else:
            self.mask_encoder = None
            self.mask_decoder = None

        # 4D RoPE setup
        self.use_4d_rope = use_4d_rope
        self.rope_dim_ratios = rope_dim_ratios
        if use_4d_rope:
            head_dim = dim // num_heads
            self.freqs_4d, self.rope_dim_split = self._build_4d_freqs(
                1024, head_dim, rope_dim_ratios
            )
        else:
            self.freqs_4d = None
            self.rope_dim_split = None

        # Replace cross attention in all blocks with LayeredCrossAttention
        for block in self.blocks:
            block.cross_attn = LayeredCrossAttention(
                dim, num_heads, (-1, -1), qk_norm, eps
            )

    def _build_4d_freqs(self, max_seq_len: int, head_dim: int, rope_dim_ratios=None):
        """
        Build 4D RoPE frequencies with L/T/H/W allocation.

        Args:
            max_seq_len: Maximum sequence length per dimension
            head_dim: Head dimension (real dimensions)
            rope_dim_ratios: (L, T, H, W) real dimension allocation, None for default

        Returns:
            freqs: [max_seq_len, head_dim//2] frequency tensor
            rope_dim_split: (l, t, h, w) complex dimension split
        """
        if rope_dim_ratios is not None:
            l_dim, t_dim, h_dim, w_dim = rope_dim_ratios
            assert l_dim + t_dim + h_dim + w_dim == head_dim, (
                f"Dimension sum {l_dim + t_dim + h_dim + w_dim} != head_dim {head_dim}"
            )
            assert all(d % 2 == 0 for d in [l_dim, t_dim, h_dim, w_dim]), (
                "All dimensions must be even"
            )
        else:
            if head_dim != 128:
                raise ValueError(
                    f"head_dim={head_dim} not supported with default rope_dim_ratios. "
                    f"Default allocation only works for head_dim=128. "
                    f"Please provide explicit rope_dim_ratios (L,T,H,W)."
                )
            # Default: L=8, T=42, H=40, W=38 (real dimensions for head_dim=128)
            l_dim = 8
            t_dim = 42
            h_dim = 40
            w_dim = head_dim - l_dim - t_dim - h_dim  # 38 for head_dim=128

        freqs = torch.cat(
            [
                rope_params(max_seq_len, l_dim),  # [max_seq_len, l_dim//2]
                rope_params(max_seq_len, t_dim),  # [max_seq_len, t_dim//2]
                rope_params(max_seq_len, h_dim),  # [max_seq_len, h_dim//2]
                rope_params(max_seq_len, w_dim),  # [max_seq_len, w_dim//2]
            ],
            dim=1,
        )

        # Return complex dimension split for rope_apply_4d
        rope_dim_split = (l_dim // 2, t_dim // 2, h_dim // 2, w_dim // 2)
        return freqs, rope_dim_split

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        prompt_lens=None,
    ):
        """
        Forward pass with LayerAdaLN modulation.

        Args:
            x: List of input tensors [C_in, F, H, W]. F = 4*T' for layered generation.
            t: Diffusion timesteps [B]
            context: Text embeddings list [L, C]
            seq_len: Maximum sequence length
            clip_fea: CLIP features for i2v mode (optional)
            y: Conditional inputs for i2v mode (optional)
            prompt_lens: [B, 3] lengths of [full, fg, bg] for LayeredCrossAttention (optional)
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # RoPE configuration
        if self.use_4d_rope:
            if self.freqs_4d.device != device:
                self.freqs_4d = self.freqs_4d.to(device)
            freqs_to_use = self.freqs_4d
            num_layers_for_rope = self.num_output_layers
            rope_dim_split = self.rope_dim_split
        else:
            freqs_to_use = self.freqs
            num_layers_for_rope = 1
            rope_dim_split = None

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len

        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # LayerAdaLN modulation
        if seq_lens.numel() > 1 and not torch.all(seq_lens == seq_lens[0]):
            raise ValueError("LayerAdaLN requires equal seq_lens across batch")
        total_tokens = seq_lens[0].item()
        tokens_per_layer = total_tokens // self.num_output_layers

        layer_mod = self.layer_adaln.get_modulation(total_tokens, tokens_per_layer, x.device)
        if layer_mod is not None and total_tokens < seq_len:
            padding = torch.zeros(
                seq_len - total_tokens, 6, self.dim,
                device=layer_mod.device, dtype=layer_mod.dtype
            )
            layer_mod = torch.cat([layer_mod, padding], dim=0)

        # Time embeddings
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Context embedding
        context_lens = None
        max_context_len = max(u.size(0) for u in context)
        effective_text_len = max(self.text_len, max_context_len)

        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(effective_text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            if prompt_lens is not None:
                raise ValueError(
                    "LayeredCrossAttention with prompt_lens is not compatible with clip_fea (I2V mode)."
                )
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # Precompute cross attention mask
        cross_attn_mask = None
        if prompt_lens is not None and hasattr(self.blocks[0].cross_attn, '_build_layer_attention_mask'):
            bool_mask = self.blocks[0].cross_attn._build_layer_attention_mask(
                batch_size=len(x),
                l_visual=seq_len,
                l_text=context.size(1),
                tokens_per_layer=tokens_per_layer,
                prompt_lens=prompt_lens,
                device=context.device
            )
            cross_attn_mask = torch.where(
                bool_mask.unsqueeze(1),
                torch.zeros(1, device=context.device, dtype=context.dtype),
                torch.full((1,), float('-inf'), device=context.device, dtype=context.dtype)
            )

        # Block arguments
        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs_to_use,
            context=context, context_lens=context_lens, num_layers=num_layers_for_rope,
            rope_dim_split=rope_dim_split, layer_mod=layer_mod,
            tokens_per_layer=tokens_per_layer, prompt_lens=prompt_lens,
            cross_attn_mask=cross_attn_mask,
        )

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    block, x, e0, seq_lens, grid_sizes, freqs_to_use, context,
                    context_lens, num_layers_for_rope, rope_dim_split, layer_mod,
                    tokens_per_layer, prompt_lens, cross_attn_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def split_layers(self, output: torch.Tensor, T_prime: int) -> dict:
        """
        Split concatenated output back into individual layers.

        Args:
            output: Output tensor [B, C, 4*T', H', W']
            T_prime: Time dimension of single layer

        Returns:
            Dictionary with 'full_video', 'background', 'foreground', 'mask' tensors
        """
        return {
            "full_video": output[:, :, 0:T_prime],
            "background": output[:, :, T_prime : 2 * T_prime],
            "foreground": output[:, :, 2 * T_prime : 3 * T_prime],
            "mask": output[:, :, 3 * T_prime : 4 * T_prime],
        }

    @classmethod
    def from_pretrained_wan(
        cls,
        pretrained_model: WanModel,
        num_output_layers: int = 4,
        mask_mode: str = "vae",
        use_4d_rope: bool = True,
        rope_dim_ratios: tuple = None,
    ):
        """
        Create LayeredWanModel from a pretrained WanModel.

        Args:
            pretrained_model: Pretrained WanModel instance
            num_output_layers: Number of layer categories
            mask_mode: Mask processing mode - "vae", "downsample", "downsample-project", "vae-project", "vae-lora", "mask-vae-project", or "mask-vae-joint"
            use_4d_rope: Whether to use 4D RoPE for (L, T, H, W) position encoding
            rope_dim_ratios: Custom dimension allocation (L, T, H, W) in real dims

        Returns:
            LayeredWanModel with weights from pretrained model
        """
        config = pretrained_model.config

        # Some attributes are in ignore_for_config, get them from model directly
        # ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']
        patch_size = getattr(pretrained_model, "patch_size", None)
        if patch_size is None:
            patch_size = pretrained_model.head.patch_size  # Get from Head module

        text_dim = getattr(pretrained_model, "text_dim", None)
        if text_dim is None:
            # Infer from text_embedding layer: Linear(text_dim, dim)
            text_dim = pretrained_model.text_embedding[0].in_features

        window_size = getattr(pretrained_model, "window_size", (-1, -1))
        qk_norm = getattr(pretrained_model, "qk_norm", True)
        cross_attn_norm = getattr(pretrained_model, "cross_attn_norm", True)

        model = cls(
            model_type=config.model_type,
            patch_size=patch_size,
            text_len=config.text_len,
            in_dim=config.in_dim,
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=text_dim,
            out_dim=config.out_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=config.eps,
            num_output_layers=num_output_layers,
            mask_mode=mask_mode,
            use_4d_rope=use_4d_rope,
            rope_dim_ratios=rope_dim_ratios,
        )

        # Copy weights from pretrained model
        pretrained_state_dict = pretrained_model.state_dict()
        model_state_dict = model.state_dict()

        # Direct key matching - no wrapper blocks anymore
        for key in pretrained_state_dict:
            if key in model_state_dict:
                # Direct match
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict)

        return model
