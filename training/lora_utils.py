# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
LoRA utilities for fine-tuning Layered Video Generation models.

Based on HuggingFace PEFT best practices:
- https://huggingface.co/docs/diffusers/en/training/lora
- https://huggingface.co/blog/lora
"""

from typing import List, Optional

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model, PeftModel

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: PEFT not installed. LoRA functionality will be limited.")


def get_lora_config(
    rank: int = 196,
    alpha: int = 392,
    use_all_linear: bool = True,
    dropout: float = 0.05,
    modules_to_save: Optional[List[str]] = None,
    mask_mode: str = "vae",
) -> "LoraConfig":
    """
    Get LoRA configuration for WanModel fine-tuning.

    Best Practices (from HuggingFace):
    - Apply LoRA to all linear layers, not just attention
    - Rank 196 provides sufficient learning capacity for complex tasks
    - Alpha typically set to 2x rank for balanced learning rate
    - Small dropout (0.05) helps prevent overfitting

    Note on patch_embedding:
    - Generally NOT fine-tuned to preserve pretrained visual features
    - Fine-tuning may harm the learned representations
    - Excluded from target_modules

    Args:
        rank: LoRA rank (recommended 32-64)
        alpha: LoRA alpha scaling factor (recommended 2x rank)
        use_all_linear: If True, apply LoRA to all linear layers (recommended)
        dropout: LoRA dropout rate
        modules_to_save: Additional modules to save fully (not LoRA)
        mask_mode: Mask processing mode - "vae", "downsample", "downsample-project", "vae-project", "mask-vae-project", or "mask-vae-joint"

    Returns:
        LoraConfig instance
    """
    if not HAS_PEFT:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    if use_all_linear:
        # Recommended: Apply to all linear layers (excluding patch_embedding)
        # Based on WanModel architecture:
        # - WanSelfAttention: q, k, v, o
        # - WanT2VCrossAttention/WanI2VCrossAttention: q, k, v, o
        # - FFN: nn.Sequential(Linear, GELU, Linear) -> ffn.0, ffn.2
        # - text_embedding: nn.Sequential(Linear, GELU, Linear) -> 0, 2
        # - time_embedding: nn.Sequential(Linear, SiLU, Linear) -> 0, 2
        # - time_projection: nn.Sequential(SiLU, Linear) -> 1
        # - Head.head: Linear
        target_modules = [
            # Self-attention layers (in WanSelfAttention)
            "self_attn.q",
            "self_attn.k",
            "self_attn.v",
            "self_attn.o",
            # Cross-attention layers (in WanT2VCrossAttention)
            "cross_attn.q",
            "cross_attn.k",
            "cross_attn.v",
            "cross_attn.o",
            # FFN layers
            "ffn.0",
            "ffn.2",
            # Output head (Head.head is a single Linear)
            "head.head",
            # Text embedding projection
            "text_embedding.0",
            "text_embedding.2",
            # Time embedding layers
            "time_embedding.0",
            "time_embedding.2",
            # Time projection (only index 1 is Linear, index 0 is SiLU)
            "time_projection.1",
        ]
    else:
        # Minimal configuration: Only attention layers
        target_modules = [
            "self_attn.q",
            "self_attn.k",
            "self_attn.v",
            "self_attn.o",
            "cross_attn.q",
            "cross_attn.k",
            "cross_attn.v",
            "cross_attn.o",
        ]

    # Default modules to save (new layers that need full training)
    # Always include layer_adaln for LayerAdaLN modulation
    # NOTE: mask_encoder/mask_decoder are NOT included here - they are managed
    # separately (like VAE) to avoid DDP conflicts when called outside model.forward()
    if modules_to_save is None:
        modules_to_save = ["layer_adaln"]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=modules_to_save if modules_to_save else None,
    )


def get_base_model(model: nn.Module) -> nn.Module:
    """Get the base model from a wrapped model (DDP, FSDP, PEFT, etc.)."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "base_model"):
        base = model.base_model
        return base.model if hasattr(base, "model") else base
    return model


def _get_model_attr(model: nn.Module, attr: str) -> Optional[nn.Module]:
    """Get an attribute from the base model, handling wrappers."""
    base = get_base_model(model)
    value = getattr(base, attr, None)
    return value


def get_mask_encoder(model: nn.Module) -> Optional[nn.Module]:
    """Get mask_encoder from model, handling DDP/FSDP/PEFT wrappers."""
    return _get_model_attr(model, "mask_encoder")


def get_mask_decoder(model: nn.Module) -> Optional[nn.Module]:
    """Get mask_decoder from model, handling DDP/FSDP/PEFT wrappers."""
    return _get_model_attr(model, "mask_decoder")


def apply_lora(model: nn.Module, config: "LoraConfig") -> nn.Module:
    """Apply LoRA to a model and configure trainable parameters."""
    if not HAS_PEFT:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, config)
    base = get_base_model(model)

    # Ensure new layers are trainable
    for attr in ["layer_adaln", "mask_encoder", "mask_decoder"]:
        module = getattr(base, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    model.print_trainable_parameters()
    return model


def get_trainable_parameters(model: nn.Module) -> dict:
    """Get statistics about trainable parameters."""
    trainable_params = 0
    all_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": 100 * trainable_params / all_params
        if all_params > 0
        else 0,
        "trainable_names": trainable_names,
    }


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save LoRA weights to disk.

    Args:
        model: Model with LoRA applied
        save_path: Path to save the weights
    """
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
    else:
        # Fallback: save only trainable parameters
        trainable_state_dict = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state_dict, f"{save_path}/lora_weights.pt")


def load_lora_weights(
    model: nn.Module, load_path: str, is_trainable: bool = False
) -> nn.Module:
    """
    Load LoRA weights from disk.

    Args:
        model: Base model to load weights into
        load_path: Path to load weights from
        is_trainable: Whether loaded adapters should be trainable (best-effort, depends on PEFT version)

    Returns:
        Model with LoRA weights loaded
    """
    if HAS_PEFT:
        try:
            try:
                model = PeftModel.from_pretrained(
                    model, load_path, is_trainable=is_trainable
                )
            except TypeError:
                model = PeftModel.from_pretrained(model, load_path)
            return model
        except Exception:
            pass

    # Fallback: load trainable parameters directly
    weights_path = f"{load_path}/lora_weights.pt"
    state_dict = torch.load(weights_path, map_location="cpu")

    # Load matching parameters
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param

    model.load_state_dict(model_state_dict)
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the base model for faster inference.

    Args:
        model: Model with LoRA applied

    Returns:
        Model with LoRA weights merged
    """
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    else:
        print(
            "Warning: Model does not support merge_and_unload. Returning original model."
        )
        return model


def load_ema_weights(model: nn.Module, ema_path: str) -> nn.Module:
    """
    Load EMA weights into a model for inference.

    EMA weights are the smoothed version of trainable parameters (LoRA + layer_adaln),
    which typically produce better generation quality than the raw training weights.

    Note: When loading for inference, PeftModel.from_pretrained may set requires_grad=False,
    so we identify trainable params by name pattern instead of requires_grad flag.

    Usage:
        # 1. Load base model and apply LoRA (or load from checkpoint)
        model = LayeredWanModel.from_pretrained_wan(base_model)
        model = load_lora_weights(model, "checkpoints/step-1000")

        # 2. Load EMA weights (replaces LoRA params with smoothed version)
        model = load_ema_weights(model, "checkpoints/step-1000/ema/ema_model.pt")

        # 3. Use for inference
        model.eval()

    Args:
        model: Model with LoRA applied (must have same structure as during training)
        ema_path: Path to ema_model.pt file

    Returns:
        Model with EMA weights loaded
    """
    # Load EMA state dict
    ema_state = torch.load(ema_path, map_location="cpu")

    # EMA state contains shadow params as a flat list
    # We need to match them to trainable parameters by order
    if "shadow_params" in ema_state:
        shadow_params = ema_state["shadow_params"]
    else:
        # Old format: state dict is the shadow params directly
        shadow_params = list(ema_state.values())

    # Get trainable parameters (same order as during training)
    # First try requires_grad (works during training), then fall back to name pattern (for inference)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # If no trainable params found, identify by name pattern (inference scenario)
    if len(trainable_params) == 0:
        trainable_params = []
        for name, param in model.named_parameters():
            # LoRA params or modules_to_save params (layer_adaln)
            if "lora_" in name or "modules_to_save" in name:
                trainable_params.append(param)

    if len(shadow_params) != len(trainable_params):
        print(
            f"Warning: EMA has {len(shadow_params)} params, model has {len(trainable_params)} trainable params"
        )
        print("Attempting to load by matching parameter count...")
        # Try to match by count - this may not work if model structure changed
        min_count = min(len(shadow_params), len(trainable_params))
        for i in range(min_count):
            if shadow_params[i].shape == trainable_params[i].shape:
                trainable_params[i].data.copy_(shadow_params[i])
            else:
                print(
                    f"  Skipping param {i}: shape mismatch {shadow_params[i].shape} vs {trainable_params[i].shape}"
                )
    else:
        # Load EMA params into trainable params
        for ema_param, model_param in zip(shadow_params, trainable_params):
            if ema_param.shape == model_param.shape:
                model_param.data.copy_(ema_param)
            else:
                print(
                    f"Warning: Shape mismatch {ema_param.shape} vs {model_param.shape}, skipping"
                )

    print(f"Loaded EMA weights from {ema_path}")
    return model


def load_checkpoint_for_inference(
    model: nn.Module,
    checkpoint_path: str,
    use_ema: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """
    Load a checkpoint for inference (convenience function).

    This combines loading LoRA weights and optionally EMA weights in one call.

    Args:
        model: Base model (before LoRA is applied)
        checkpoint_path: Path to checkpoint directory (e.g., "outputs/checkpoints/step-1000")
        use_ema: Whether to use EMA weights (recommended for better quality)
        device: Device to load model to

    Returns:
        Model ready for inference

    Example:
        from wan.modules.model import WanModel
        from wan.modules.layered_model import LayeredWanModel
        from training.lora_utils import load_checkpoint_for_inference

        # Load base model
        base_model = WanModel.from_pretrained("path/to/wan2.1")
        model = LayeredWanModel.from_pretrained_wan(base_model)

        # Load checkpoint (with EMA for better quality)
        model = load_checkpoint_for_inference(
            model,
            "outputs/checkpoints/step-5000",
            use_ema=True
        )
        model.eval()
    """
    import os

    # Load LoRA weights
    model = load_lora_weights(model, checkpoint_path, is_trainable=False)

    # Load EMA weights if available and requested
    if use_ema:
        ema_path = os.path.join(checkpoint_path, "ema", "ema_model.pt")
        if os.path.exists(ema_path):
            model = load_ema_weights(model, ema_path)
        else:
            print(f"EMA weights not found at {ema_path}, using regular weights")

    model = model.to(device)
    model.eval()

    return model


def _unfreeze_module(model: nn.Module, attr: str) -> int:
    """Unfreeze a module's parameters and return param count."""
    base = get_base_model(model)
    module = getattr(base, attr, None)
    if module is None:
        return 0
    param_count = 0
    for param in module.parameters():
        param.requires_grad = True
        param_count += param.numel()
    return param_count


def unfreeze_layer_adaln(model: nn.Module):
    """Ensure LayerAdaLN parameters are trainable."""
    count = _unfreeze_module(model, "layer_adaln")
    if count > 0:
        print(f"LayerAdaLN parameters unfrozen: {count} params")


def unfreeze_mask_encoder_decoder(model: nn.Module, mask_mode: str = "vae"):
    """Ensure MaskEncoder/Decoder are trainable (downsample-project mode only)."""
    if mask_mode != "downsample-project":
        return
    for attr in ["mask_encoder", "mask_decoder"]:
        count = _unfreeze_module(model, attr)
        if count > 0:
            print(f"{attr} parameters unfrozen: {count} params")


class LoRALinear(nn.Module):
    """
    Manual LoRA implementation for environments without PEFT.

    This is a fallback implementation when PEFT is not available.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear
        result = self.linear(x)

        # LoRA addition
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into the original linear layer."""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling


def apply_lora_manual(
    model: nn.Module,
    rank: int = 32,
    alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA manually without PEFT library.

    This is a fallback when PEFT is not available.

    Args:
        model: Model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: List of module name patterns to apply LoRA to
        dropout: Dropout rate

    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ["q", "k", "v", "o"]

    # Find and replace linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should have LoRA
            should_apply = any(target in name for target in target_modules)
            if should_apply:
                # Get parent module and attribute name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    attr_name = name

                # Replace with LoRA version
                lora_linear = LoRALinear(module, rank, alpha, dropout)
                setattr(parent, attr_name, lora_linear)

    return model
