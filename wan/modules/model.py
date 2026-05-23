# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ["WanModel", "clear_rope_4d_cache"]

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


_ROPE_4D_FREQS_CACHE: dict = {}


def _get_cached_freqs_4d(
    num_layers: int,
    t: int,
    h: int,
    w: int,
    freqs_l: torch.Tensor,
    freqs_t: torch.Tensor,
    freqs_h: torch.Tensor,
    freqs_w: torch.Tensor,
    rope_dim_split: tuple = None,
) -> torch.Tensor:
    # Include rope_dim_split and dtype in cache key to avoid stale cache
    # when config changes (e.g., different rope_dim_ratios or theta)
    cache_key = (num_layers, t, h, w, freqs_l.device, freqs_l.dtype, rope_dim_split)
    if cache_key not in _ROPE_4D_FREQS_CACHE:
        seq_len = num_layers * t * h * w
        freqs_i = (
            torch.cat(
                [
                    freqs_l[:num_layers]
                    .view(num_layers, 1, 1, 1, -1)
                    .expand(num_layers, t, h, w, -1),
                    freqs_t[:t].view(1, t, 1, 1, -1).expand(num_layers, t, h, w, -1),
                    freqs_h[:h].view(1, 1, h, 1, -1).expand(num_layers, t, h, w, -1),
                    freqs_w[:w].view(1, 1, 1, w, -1).expand(num_layers, t, h, w, -1),
                ],
                dim=-1,
            )
            .contiguous()
            .reshape(seq_len, 1, -1)
        )
        _ROPE_4D_FREQS_CACHE[cache_key] = freqs_i
    return _ROPE_4D_FREQS_CACHE[cache_key]


def clear_rope_4d_cache():
    _ROPE_4D_FREQS_CACHE.clear()


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_4d(x, grid_sizes, freqs, num_layers=4, rope_dim_split=None):
    """
    Apply 4D RoPE for (Layer, Time, Height, Width).

    Args:
        x: [B, seq_len, num_heads, head_dim]
        grid_sizes: [B, 3] - (F_total, H, W) where F_total = num_layers * T
        freqs: [max_seq_len, head_dim//2] - precomputed 4D frequencies
        num_layers: int - number of layers
        rope_dim_split: tuple (l, t, h, w) - complex dimension split, None for default

    Position encoding: each token gets position encoding based on (layer_idx, t, h, w)
    """
    n, c = x.size(2), x.size(3) // 2
    max_seq_len = freqs.size(0)
    original_dtype = x.dtype

    if rope_dim_split is not None:
        l_dim, t_dim, h_dim, w_dim = rope_dim_split
    else:
        l_dim = 4
        t_dim = 21
        h_dim = 20
        w_dim = c - l_dim - t_dim - h_dim

    freqs_l, freqs_t, freqs_h, freqs_w = freqs.split(
        [l_dim, t_dim, h_dim, w_dim], dim=1
    )

    output = []
    for i, (f_total, h, w) in enumerate(grid_sizes.tolist()):
        assert f_total % num_layers == 0, (
            f"f_total ({f_total}) must be divisible by num_layers ({num_layers}). "
            f"Check that your input has exactly {num_layers} concatenated layers."
        )
        t = f_total // num_layers

        max_dim = max(num_layers, t, h, w)
        if max_dim > max_seq_len:
            raise ValueError(
                f"RoPE freqs overflow: max(num_layers={num_layers}, t={t}, h={h}, w={w}) = {max_dim} "
                f"exceeds max_seq_len={max_seq_len}. Increase max_seq_len in _build_4d_freqs()."
            )

        seq_len = f_total * h * w

        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )

        freqs_i = _get_cached_freqs_4d(
            num_layers,
            t,
            h,
            w,
            freqs_l,
            freqs_t,
            freqs_h,
            freqs_w,
            rope_dim_split=(l_dim, t_dim, h_dim, w_dim),
        )

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).to(original_dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self, x, seq_lens, grid_sizes, freqs, num_layers=1, rope_dim_split=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            num_layers(int): Number of layers for 4D RoPE. If > 1, uses rope_apply_4d
            rope_dim_split(tuple): (l, t, h, w) complex dimension split for 4D RoPE
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Apply RoPE (3D or 4D based on num_layers)
        if num_layers > 1:
            q_rope = rope_apply_4d(q, grid_sizes, freqs, num_layers, rope_dim_split)
            k_rope = rope_apply_4d(k, grid_sizes, freqs, num_layers, rope_dim_split)
        else:
            q_rope = rope_apply(q, grid_sizes, freqs)
            k_rope = rope_apply(k, grid_sizes, freqs)

        x = flash_attention(
            q=q_rope, k=k_rope, v=v, k_lens=seq_lens, window_size=self.window_size
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class LayeredCrossAttention(WanSelfAttention):
    """
    Cross-attention with layer-aware masking for multi-prompt scenarios.

    Supports different visual token groups (full_video, background, foreground, mask)
    attending to different subsets of concatenated text embeddings [full | fg | bg].
    """

    # Layer-prompt visibility: [num_layers=4, num_prompts=3]
    # Prompts order: [full, fg, bg]
    # Layers: 0=full_video, 1=background, 2=foreground, 3=mask
    LAYER_PROMPT_VISIBILITY = [
        (True, True, True),   # full_video: sees all prompts
        (True, False, True),  # background: sees full + bg (not fg)
        (True, True, False),  # foreground: sees full + fg (not bg)
        (True, True, False),  # mask: sees full + fg (not bg, same as foreground)
    ]

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

    def forward(
        self,
        x,                    # [B, L_visual, C]
        context,              # [B, L_text_total, C] = [full | fg | bg]
        context_lens,         # [B] total context lengths (can be None)
        tokens_per_layer=None,  # int: number of visual tokens per layer
        prompt_lens=None,     # [B, 3] lengths of [full, fg, bg] for each batch
        cross_attn_mask=None,  # [B, L_visual, L_text] precomputed mask (optional)
    ):
        """
        Args:
            x: Visual tokens [B, L_visual, C]
            context: Concatenated text embeddings [B, L_text, C]
            context_lens: Total context lengths per batch [B] (optional, for compatibility)
            tokens_per_layer: Visual tokens per layer category
            prompt_lens: Lengths of each prompt type [B, 3]
            cross_attn_mask: Precomputed attention mask [B, L_visual, L_text] (optional)
                If provided, tokens_per_layer and prompt_lens are ignored.
        """
        # Fallback to standard cross attention if no layered params provided
        if cross_attn_mask is None and (tokens_per_layer is None or prompt_lens is None):
            return self._standard_cross_attention(x, context, context_lens)

        b, n, d = x.size(0), self.num_heads, self.head_dim
        l_visual = x.size(1)
        l_text = context.size(1)

        # Compute Q, K, V
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # Use precomputed mask if provided, otherwise build it
        if cross_attn_mask is not None:
            attn_mask = cross_attn_mask
        else:
            attn_mask = self._build_layer_attention_mask(
                b, l_visual, l_text, tokens_per_layer, prompt_lens, x.device
            )

        # Apply attention with mask using PyTorch SDPA
        x = self._masked_attention(q, k, v, attn_mask)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x

    def _standard_cross_attention(self, x, context, context_lens):
        """Fallback to standard WanT2VCrossAttention behavior."""
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        x = self.o(x)
        return x

    def _build_layer_attention_mask(
        self, batch_size, l_visual, l_text, tokens_per_layer, prompt_lens, device
    ):
        """
        Build attention mask based on layer-prompt visibility.

        Returns:
            mask: [B, L_visual, L_text] where True = attend, False = mask out
        """
        num_layers = 4
        mask = torch.zeros(batch_size, l_visual, l_text, dtype=torch.bool, device=device)

        for b_idx in range(batch_size):
            lens = [prompt_lens[b_idx, i].item() for i in range(3)]
            total_valid_text = sum(lens)

            for layer_idx in range(num_layers):
                v_start = layer_idx * tokens_per_layer
                v_end = min((layer_idx + 1) * tokens_per_layer, l_visual)
                if v_start >= l_visual:
                    break

                visibility = self.LAYER_PROMPT_VISIBILITY[layer_idx]
                text_pos = 0
                for prompt_idx, prompt_len in enumerate(lens):
                    if visibility[prompt_idx] and prompt_len > 0:
                        mask[b_idx, v_start:v_end, text_pos:text_pos + prompt_len] = True
                    text_pos += prompt_len

            # Padding visual tokens attend to all valid text to avoid NaN
            padding_start = num_layers * tokens_per_layer
            if padding_start < l_visual and total_valid_text > 0:
                mask[b_idx, padding_start:, :total_valid_text] = True

        return mask

    def _masked_attention(self, q, k, v, attn_mask):
        """Apply attention with mask using PyTorch SDPA."""
        # Transpose for SDPA: [B, N, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Convert bool mask to float if needed
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L_q, L_k]
            attn_mask = torch.where(
                attn_mask,
                torch.zeros(1, device=q.device, dtype=q.dtype),
                torch.full((1,), float('-inf'), device=q.device, dtype=q.dtype)
            )
        else:
            attn_mask = attn_mask.to(q.dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )
        return out.transpose(1, 2).contiguous()


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
    "layered_cross_attn": LayeredCrossAttention,
}


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        num_layers=1,
        rope_dim_split=None,
        layer_mod=None,
        tokens_per_layer=None,
        prompt_lens=None,
        cross_attn_mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            num_layers(int): Number of layers for 4D RoPE
            rope_dim_split(tuple): (l, t, h, w) complex dimension split for 4D RoPE
            layer_mod(Tensor, optional): Shape [L, 6, C] layer modulation from LayerAdaLN
            tokens_per_layer(int, optional): Visual tokens per layer for LayeredCrossAttention
            prompt_lens(Tensor, optional): Shape [B, 3] for LayeredCrossAttention
            cross_attn_mask(Tensor, optional): Precomputed mask [B, L_visual, L_text]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)  # 6 x [B, 1, C]

            # If layer_mod is provided, merge it with timestep modulation
            # layer_mod: [L, 6, C] -> add to e[i] to get per-token modulation
            if layer_mod is not None:
                # Convert layer_mod to float32 for AdaLN computation
                layer_mod_f32 = layer_mod.float()
                # Split layer_mod into 6 chunks: 6 x [L, 1, C]
                layer_mod_chunks = layer_mod_f32.chunk(6, dim=1)
                # Combine: e[i] is [B, 1, C], layer_mod_chunks[i] is [L, 1, C]
                # Result: [B, L, C] (broadcast B, expand L)
                e = [
                    e[i] + layer_mod_chunks[i].squeeze(1).unsqueeze(0)
                    for i in range(6)
                ]
                # Now e[i] is [B, L, C] for per-token modulation

        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs,
            num_layers,
            rope_dim_split,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, tokens_per_layer, prompt_lens, cross_attn_mask):
            # Support both standard and layered cross attention
            if isinstance(self.cross_attn, LayeredCrossAttention):
                x = x + self.cross_attn(
                    self.norm3(x), context, context_lens,
                    tokens_per_layer=tokens_per_layer,
                    prompt_lens=prompt_lens,
                    cross_attn_mask=cross_attn_mask
                )
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e, tokens_per_layer, prompt_lens, cross_attn_mask)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280)
            )

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v", "vace"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

        # gradient checkpointing (disabled by default)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, enable: bool = True):
        """
        Enable or disable gradient checkpointing for memory efficiency.

        When enabled, intermediate activations are recomputed during backward
        pass instead of being stored, reducing memory usage at the cost of
        ~30% slower training.

        Args:
            enable: Whether to enable gradient checkpointing
        """
        self.gradient_checkpointing = enable

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                # Pass args positionally to match block.forward signature:
                # forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)
                x = checkpoint(
                    block,
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
