# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
"""
Layered Text-to-Video Generation Pipeline.

Generates multi-layer video outputs:
- full_video: Complete video
- background: Background layer
- foreground: Foreground RGB layer
- mask: Foreground alpha mask
"""

import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.layered_model import LayeredWanModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class LayeredWanT2V:
    """
    Layered Text-to-Video generation pipeline.

    Generates multi-layer video outputs from text prompts:
    - full_video: Complete video
    - background: Background layer
    - foreground: Foreground RGB layer
    - mask: Foreground alpha mask

    The model processes input where layers are concatenated along the time dimension.
    """

    def __init__(
        self,
        config,
        checkpoint_dir: str,
        lora_path: Optional[str] = None,
        use_ema: bool = False,
        mask_mode: str = "vae",
        use_4d_rope: Optional[bool] = None,
        rope_dim_ratios: Optional[Tuple[int, int, int, int]] = None,
        device_id: int = 0,
        rank: int = 0,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        t5_cpu: bool = False,
        mask_vae_path: Optional[str] = None,
        mask_vae_proj_path: Optional[str] = None,
        mask_vae_lora_path: Optional[str] = None,
    ):
        """
        Initialize the LayeredWanT2V pipeline.

        Args:
            config: Model configuration (EasyDict)
            checkpoint_dir: Path to Wan2.1 checkpoint directory
            lora_path: Path to LoRA weights (optional)
            use_ema: Use EMA weights for inference (recommended for better quality)
            mask_mode: Mask processing mode ("vae", "downsample", "downsample-project", "vae-project", "vae-lora", "mask-vae-project", or "mask-vae-joint")
            use_4d_rope: Use 4D RoPE (L, T, H, W) instead of 3D. None = use config default
            rope_dim_ratios: Custom 4D RoPE dimension allocation (L, T, H, W). None = use config default
            device_id: GPU device ID
            rank: Process rank for distributed inference
            t5_fsdp: Enable FSDP for T5
            dit_fsdp: Enable FSDP for DiT
            t5_cpu: Place T5 on CPU
            mask_vae_path: Path to MaskVAE checkpoint (mask-vae-project/mask-vae-joint mode)
            mask_vae_proj_path: Path to projection layers (vae-project/mask-vae-project/mask-vae-joint mode)
            mask_vae_lora_path: Path to VAE LoRA checkpoint (vae-lora mode)
        """
        # Get 4D RoPE settings from config if not explicitly provided
        if use_4d_rope is None:
            use_4d_rope = getattr(config, "use_4d_rope", True)
        if rope_dim_ratios is None:
            rope_dim_ratios = getattr(config, "rope_dim_ratios", None)
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.lora_path = lora_path
        self.use_ema = use_ema
        self.mask_mode = mask_mode
        self.mask_vae = None
        self.mask_vae_proj_in = None
        self.mask_vae_proj_out = None
        self.mask_vae_lora_decoder = None
        self.mask_vae_path = mask_vae_path
        self.mask_vae_proj_path = mask_vae_proj_path
        self.mask_vae_lora_path = mask_vae_lora_path

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # T5 Text Encoder
        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        # VAE
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae_checkpoint_path = os.path.join(checkpoint_dir, config.vae_checkpoint)
        self.vae = WanVAE(
            vae_pth=self.vae_checkpoint_path,
            device=self.device,
        )

        # DiT Model
        rope_str = f"use_4d_rope={use_4d_rope}, rope_dim_ratios={rope_dim_ratios}"
        logging.info(
            f"Creating LayeredWanModel from {checkpoint_dir} (mask_mode={mask_mode}, {rope_str})"
        )
        base_model = WanModel.from_pretrained(checkpoint_dir)
        self.model = LayeredWanModel.from_pretrained_wan(
            base_model,
            num_output_layers=4,
            mask_mode=mask_mode,
            use_4d_rope=use_4d_rope,
            rope_dim_ratios=rope_dim_ratios,
        )
        del base_model

        # Load LoRA weights if provided
        if lora_path:
            self._load_lora_weights(lora_path, use_ema=use_ema)

        self.model.eval().requires_grad_(False)

        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        # Ensure mask modules are on correct device after FSDP sharding
        # FSDP only shards modules in model.blocks, so mask_encoder/decoder stay on CPU
        self._ensure_mask_modules_on_device()

        # Load projection layers for vae-project/mask-vae-project/mask-vae-joint mode
        if self.mask_mode == "vae-project":
            self._load_vae_project()
        elif self.mask_mode == "vae-lora":
            self._load_vae_lora()
        elif self.mask_mode in ("mask-vae-project", "mask-vae-joint"):
            self._load_mask_vae()

        self.sample_neg_prompt = config.sample_neg_prompt

    def _load_lora_weights(self, lora_path: str, use_ema: bool = False):
        """Load LoRA weights from checkpoint.

        Args:
            lora_path: Path to checkpoint directory containing LoRA weights
            use_ema: If True, load EMA weights (smoother, often better quality)
        """
        logging.info(f"Loading LoRA weights from {lora_path}")

        try:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_path)

            # Load EMA weights if requested
            if use_ema:
                ema_path = os.path.join(lora_path, "ema", "ema_model.pt")
                if os.path.exists(ema_path):
                    self._load_ema_into_model(ema_path)
                    logging.info("EMA weights loaded successfully")
                else:
                    logging.warning(
                        f"EMA weights not found at {ema_path}, using regular weights"
                    )

            self.model = self.model.merge_and_unload()
            logging.info("LoRA weights loaded and merged successfully")
        except ImportError:
            # Fallback: load trainable weights directly
            weights_path = os.path.join(lora_path, "lora_weights.pt")
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                model_state_dict = self.model.state_dict()
                for name, param in state_dict.items():
                    if name in model_state_dict:
                        model_state_dict[name] = param
                self.model.load_state_dict(model_state_dict)
                logging.info("LoRA weights loaded (fallback method)")
            else:
                logging.warning(f"LoRA weights not found at {weights_path}")

        # Load mask_encoder/decoder if saved separately (downsample-project mode)
        self._load_mask_modules(lora_path)

    def _load_mask_modules(self, lora_path: str):
        """Load separately saved mask_encoder/decoder for downsample-project mode."""
        from .modules.layered_model import MaskEncoder, MaskDecoder

        mask_enc_path = os.path.join(lora_path, "mask_encoder.pt")
        mask_dec_path = os.path.join(lora_path, "mask_decoder.pt")

        if self.mask_mode != "downsample-project":
            return

        if os.path.exists(mask_enc_path):
            if self.model.mask_encoder is None:
                self.model.mask_encoder = MaskEncoder(
                    in_channels=1, out_channels=16, hidden_channels=32
                )
            self.model.mask_encoder.load_state_dict(
                torch.load(mask_enc_path, map_location="cpu")
            )
            self.model.mask_encoder.to(self.device)
            self.model.mask_encoder.eval()
            logging.info(f"Loaded mask_encoder from {mask_enc_path}")
        elif self.model.mask_encoder is not None:
            # Ensure mask_encoder is on correct device even without checkpoint
            self.model.mask_encoder.to(self.device)
            self.model.mask_encoder.eval()

        if os.path.exists(mask_dec_path):
            if self.model.mask_decoder is None:
                self.model.mask_decoder = MaskDecoder(
                    in_channels=16, out_channels=1, hidden_channels=32
                )
            self.model.mask_decoder.load_state_dict(
                torch.load(mask_dec_path, map_location="cpu")
            )
            self.model.mask_decoder.to(self.device)
            self.model.mask_decoder.eval()
            logging.info(f"Loaded mask_decoder from {mask_dec_path}")
        elif self.model.mask_decoder is not None:
            # Ensure mask_decoder is on correct device even without checkpoint
            self.model.mask_decoder.to(self.device)
            self.model.mask_decoder.eval()

    def _ensure_mask_modules_on_device(self):
        """Ensure mask_encoder/decoder are on correct device.

        This is called after FSDP sharding since FSDP only shards modules in
        model.blocks, leaving mask_encoder/decoder on CPU.
        """
        if self.mask_mode != "downsample-project":
            return

        # Access underlying model for FSDP-wrapped models
        model = getattr(self.model, "module", self.model)

        if model.mask_encoder is not None:
            model.mask_encoder.to(self.device)
            model.mask_encoder.eval()

        if model.mask_decoder is not None:
            model.mask_decoder.to(self.device)
            model.mask_decoder.eval()

    def _load_vae_project(self):
        """Load projection layers for vae-project mode."""
        from .modules.mask_vae_project import load_mask_latent_projects

        if self.mask_vae_proj_path:
            if not os.path.exists(self.mask_vae_proj_path):
                raise FileNotFoundError(
                    f"--mask_vae_proj_path not found: {self.mask_vae_proj_path}"
                )
            self.mask_vae_proj_in, self.mask_vae_proj_out = load_mask_latent_projects(
                self.mask_vae_proj_path, device=self.device
            )
            logging.info(f"Loaded projection layers from {self.mask_vae_proj_path}")
        elif self.lora_path:
            proj_path = os.path.join(self.lora_path, "mask_vae_projects.pt")
            if os.path.exists(proj_path):
                self.mask_vae_proj_in, self.mask_vae_proj_out = (
                    load_mask_latent_projects(proj_path, device=self.device)
                )
                logging.info(f"Loaded projection layers from {proj_path}")
            else:
                raise FileNotFoundError(
                    f"vae-project mode requires mask_vae_projects.pt but not found at {proj_path}. "
                    f"Please train with vae-project mode first."
                )
        else:
            raise ValueError(
                "vae-project mode requires --mask_vae_proj_path or mask_vae_projects.pt in --lora_path."
            )

        self.mask_vae_proj_in.eval()
        self.mask_vae_proj_out.eval()

    def _load_vae_lora(self):
        """Load VAE decoder LoRA for vae-lora mode."""
        from .modules.mask_vae_lora import load_mask_vae_lora_state, build_decoder_from_state

        if self.mask_vae_lora_path:
            if not os.path.exists(self.mask_vae_lora_path):
                raise FileNotFoundError(
                    f"--mask_vae_lora_path not found: {self.mask_vae_lora_path}"
                )
            lora_path = self.mask_vae_lora_path
        elif self.lora_path:
            lora_path = os.path.join(self.lora_path, "mask_vae_lora.pt")
            if not os.path.exists(lora_path):
                raise FileNotFoundError(
                    f"vae-lora mode requires mask_vae_lora.pt but not found at {lora_path}. "
                    f"Please provide --mask_vae_lora_path or include mask_vae_lora.pt in --lora_path."
                )
        else:
            raise ValueError(
                "vae-lora mode requires --mask_vae_lora_path or mask_vae_lora.pt in --lora_path."
            )

        state = load_mask_vae_lora_state(lora_path, device=self.device)
        self.mask_vae_lora_decoder = build_decoder_from_state(
            state,
            vae_pth=self.vae_checkpoint_path,
            device=self.device,
            dtype=self.vae.dtype,
        )
        logging.info(f"Loaded Mask VAE LoRA decoder from {lora_path}")

    def _load_mask_vae(self):
        """Load MaskVAE and project layers for mask-vae-project/mask-vae-joint mode."""
        from .modules.mask_vae import load_mask_vae
        from .modules.mask_vae_project import load_mask_vae_projects

        if self.mask_vae_path:
            if not os.path.exists(self.mask_vae_path):
                raise FileNotFoundError(
                    f"--mask_vae_path not found: {self.mask_vae_path}"
                )
            self.mask_vae = load_mask_vae(self.mask_vae_path, device=self.device)
            logging.info(f"Loaded MaskVAE from {self.mask_vae_path}")
        elif self.lora_path:
            vae_path = os.path.join(self.lora_path, "mask_vae.pt")
            if os.path.exists(vae_path):
                self.mask_vae = load_mask_vae(vae_path, device=self.device)
                logging.info(f"Loaded MaskVAE from {vae_path}")
            else:
                raise FileNotFoundError(
                    f"mask-vae-project/mask-vae-joint mode requires mask_vae.pt but not found at {vae_path}. "
                    f"Please provide --mask_vae_path or include mask_vae.pt in --lora_path."
                )
        else:
            raise ValueError(
                "mask-vae-project/mask-vae-joint mode requires --mask_vae_path or mask_vae.pt in --lora_path."
            )
        self.mask_vae.eval()

        if self.mask_vae_proj_path:
            if not os.path.exists(self.mask_vae_proj_path):
                raise FileNotFoundError(
                    f"--mask_vae_proj_path not found: {self.mask_vae_proj_path}"
                )
            self.mask_vae_proj_in, self.mask_vae_proj_out = load_mask_vae_projects(
                self.mask_vae_proj_path, device=self.device
            )
            logging.info(f"Loaded MaskVAE projects from {self.mask_vae_proj_path}")
        elif self.lora_path:
            proj_path = os.path.join(self.lora_path, "mask_vae_projects.pt")
            if os.path.exists(proj_path):
                self.mask_vae_proj_in, self.mask_vae_proj_out = load_mask_vae_projects(
                    proj_path, device=self.device
                )
                logging.info(f"Loaded MaskVAE projects from {proj_path}")
            else:
                raise FileNotFoundError(
                    f"mask-vae-project/mask-vae-joint mode requires mask_vae_projects.pt but not found at {proj_path}. "
                    f"Please provide --mask_vae_proj_path or include mask_vae_projects.pt in --lora_path."
                )
        else:
            raise ValueError(
                "mask-vae-project/mask-vae-joint mode requires --mask_vae_proj_path or mask_vae_projects.pt in --lora_path."
            )
        self.mask_vae_proj_in.eval()
        self.mask_vae_proj_out.eval()

    def _load_ema_into_model(self, ema_path: str):
        """Load EMA weights into model's trainable parameters.

        Note: When loading for inference, PeftModel.from_pretrained sets
        requires_grad=False, so we identify trainable params by name pattern
        (lora_ or modules_to_save) instead of requires_grad flag.

        Args:
            ema_path: Path to ema_model.pt file
        """
        ema_state = torch.load(ema_path, map_location="cpu")

        # EMA state contains shadow params
        if "shadow_params" in ema_state:
            shadow_params = ema_state["shadow_params"]
        else:
            shadow_params = list(ema_state.values())

        # Get parameters that were trainable during training by name pattern
        # (requires_grad may be False during inference loading)
        trainable_params = []
        for name, param in self.model.named_parameters():
            # LoRA params or modules_to_save params (layer_adaln)
            if "lora_" in name or "modules_to_save" in name:
                trainable_params.append(param)

        if len(shadow_params) != len(trainable_params):
            logging.warning(
                f"EMA has {len(shadow_params)} params, model has {len(trainable_params)} trainable params"
            )
            min_count = min(len(shadow_params), len(trainable_params))
            for i in range(min_count):
                if shadow_params[i].shape == trainable_params[i].shape:
                    trainable_params[i].data.copy_(shadow_params[i])
        else:
            for ema_param, model_param in zip(shadow_params, trainable_params):
                if ema_param.shape == model_param.shape:
                    model_param.data.copy_(ema_param)

    def format_layered_prompt(
        self,
        full_prompt: str,
        fg_prompt: str = "",
        bg_prompt: str = "",
    ) -> str:
        """
        Format prompts for multi-layer generation.

        Args:
            full_prompt: Description of the full video
            fg_prompt: Description of the foreground (optional)
            bg_prompt: Description of the background (optional)

        Returns:
            Formatted prompt with layer tags
        """
        if not fg_prompt:
            fg_prompt = "foreground object"
        if not bg_prompt:
            bg_prompt = "background scene"

        prompt = (
            f"<full>{full_prompt}</full>"
            f"<bg>{bg_prompt}</bg>"
            f"<fg>{fg_prompt}</fg>"
            f"<mask>foreground alpha mask</mask>"
        )
        return prompt

    def generate(
        self,
        input_prompt: str,
        fg_prompt: str = "",
        bg_prompt: str = "",
        size: Tuple[int, int] = (1280, 720),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multi-layer video from text prompt.

        Args:
            input_prompt: Text prompt for the full video
            fg_prompt: Text prompt for foreground (optional)
            bg_prompt: Text prompt for background (optional)
            size: Output video resolution (width, height)
            frame_num: Number of frames (should be 4n+1)
            shift: Noise schedule shift parameter
            sample_solver: Sampler type ("unipc" or "dpm++")
            sampling_steps: Number of diffusion steps
            guide_scale: Classifier-free guidance scale
            n_prompt: Negative prompt
            seed: Random seed (-1 for random)
            offload_model: Offload models to CPU during generation

        Returns:
            Dictionary with:
                - full_video: [3, T, H, W] Complete video
                - background: [3, T, H, W] Background layer
                - foreground: [3, T, H, W] Foreground RGB
                - mask: [1, T, H, W] Foreground alpha mask (range [0, 1])
        """
        # Calculate target shape
        F = frame_num
        T_prime = (F - 1) // self.vae_stride[0] + 1
        H_prime = size[1] // self.vae_stride[1]
        W_prime = size[0] // self.vae_stride[2]

        # 4x T dimension for concatenated layers
        target_shape = (self.vae.model.z_dim, 4 * T_prime, H_prime, W_prime)

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
        )

        # Handle negative prompt
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Set seed
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Text encoding for LayeredCrossAttention
        full_prompt = input_prompt
        foreground_prompt = fg_prompt or "foreground object"
        background_prompt = bg_prompt or "background scene"

        def encode_prompts(prompts, enc_device):
            """Encode list of prompts and return concatenated context + lens."""
            encodings = [self.text_encoder([p], enc_device)[0] for p in prompts]
            combined = torch.cat(encodings, dim=0)
            if enc_device == torch.device("cpu"):
                combined = combined.to(self.device)
            lens = torch.tensor(
                [[len(e) for e in encodings]], device=self.device, dtype=torch.long
            )
            return [combined], lens

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context, prompt_lens = encode_prompts(
                [full_prompt, foreground_prompt, background_prompt], self.device
            )
            # Negative prompt replicated 3x
            ctx_null = self.text_encoder([n_prompt], self.device)[0]
            context_null = [torch.cat([ctx_null] * 3, dim=0)]
            prompt_lens_null = torch.tensor(
                [[len(ctx_null)] * 3], device=self.device, dtype=torch.long
            )
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context, prompt_lens = encode_prompts(
                [full_prompt, foreground_prompt, background_prompt], torch.device("cpu")
            )
            ctx_null = self.text_encoder([n_prompt], torch.device("cpu"))[0]
            context_null = [torch.cat([ctx_null] * 3, dim=0).to(self.device)]
            prompt_lens_null = torch.tensor(
                [[len(ctx_null)] * 3], device=self.device, dtype=torch.long
            )

        # Initialize noise
        noise = [torch.randn(*target_shape, dtype=torch.float32, device=self.device, generator=seed_g)]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # Sampling
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Setup scheduler
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas,
                )
            else:
                raise NotImplementedError(f"Unsupported solver: {sample_solver}")

            # Sampling loop
            latents = noise

            arg_c = {"context": context, "seq_len": seq_len, "prompt_lens": prompt_lens}
            arg_null = {"context": context_null, "seq_len": seq_len, "prompt_lens": prompt_lens_null}

            for t in tqdm(timesteps, desc="Sampling"):
                latent_model_input = latents
                timestep = torch.stack([t])

                # For vae-project mode: apply project_in to mask slice before model
                if (
                    self.mask_mode == "vae-project"
                    and self.mask_vae_proj_in is not None
                ):
                    lat = latent_model_input[0]  # [16, 4*T', H', W']
                    mask_slice = lat[:, 3 * T_prime : 4 * T_prime]  # [16, T', H', W']
                    mask_slice_proj = self.mask_vae_proj_in(mask_slice)
                    lat = torch.cat(
                        [
                            lat[:, 0 : 3 * T_prime],
                            mask_slice_proj,
                        ],
                        dim=1,
                    )
                    latent_model_input = [lat]

                self.model.to(self.device)
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null
                )[0]

                # For vae-project mode: apply project_out to mask slice of predictions
                if (
                    self.mask_mode == "vae-project"
                    and self.mask_vae_proj_out is not None
                ):
                    # Apply to cond prediction
                    mask_v_cond = noise_pred_cond[:, 3 * T_prime : 4 * T_prime]
                    mask_v_cond_proj = self.mask_vae_proj_out(mask_v_cond)
                    noise_pred_cond = torch.cat(
                        [
                            noise_pred_cond[:, 0 : 3 * T_prime],
                            mask_v_cond_proj,
                        ],
                        dim=1,
                    )
                    # Apply to uncond prediction
                    mask_v_uncond = noise_pred_uncond[:, 3 * T_prime : 4 * T_prime]
                    mask_v_uncond_proj = self.mask_vae_proj_out(mask_v_uncond)
                    noise_pred_uncond = torch.cat(
                        [
                            noise_pred_uncond[:, 0 : 3 * T_prime],
                            mask_v_uncond_proj,
                        ],
                        dim=1,
                    )

                # CFG
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                # Scheduler step
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents[0]  # [16, 4*T', H', W']

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            # Split and decode layers
            if self.rank == 0:
                # target_size = (T, H, W) for mask upsampling in downsample mode
                target_size = (frame_num, size[1], size[0])
                results = self._decode_layers(x0, T_prime, target_size)
            else:
                results = None

        # Cleanup
        del noise, latents, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return results

    def _decode_layers(
        self, x0: torch.Tensor, T_prime: int, target_size: Tuple[int, int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Split concatenated latent and decode each layer.

        Args:
            x0: Concatenated latent [16, 4*T', H', W']
            T_prime: Time dimension of single layer
            target_size: Target video size (T, H, W) for mask upsampling in downsample mode

        Returns:
            Dictionary with decoded video layers
        """
        # Split along T dimension
        full_video_z = x0[:, 0:T_prime]
        background_z = x0[:, T_prime : 2 * T_prime]
        foreground_z = x0[:, 2 * T_prime : 3 * T_prime]
        mask_z = x0[:, 3 * T_prime : 4 * T_prime]

        # Decode each layer
        full_video = self.vae.decode([full_video_z])[0]  # [3, T, H, W]
        background = self.vae.decode([background_z])[0]  # [3, T, H, W]
        foreground = self.vae.decode([foreground_z])[0]  # [3, T, H, W]

        # Decode mask based on mask_mode
        if self.mask_mode == "vae":
            # Original: decode through VAE
            mask_3ch = self.vae.decode([mask_z])[0]  # [3, T, H, W]
            # Convert mask from [-1, 1] to [0, 1] and average channels
            mask = (mask_3ch.mean(dim=0, keepdim=True) + 1) / 2  # [1, T, H, W]
            mask = mask.clamp(0, 1)
        elif self.mask_mode == "downsample":
            # Average channels + upsample
            # Training: mask [0,1] -> *2-1 -> [-1,1] -> repeat(16) -> mask_z
            # Inference: mask_z -> mean(16ch) -> (+1)/2 -> [0,1] -> upsample -> mask
            mask_down = mask_z.mean(
                dim=0, keepdim=True
            )  # [16, T', H', W'] -> [1, T', H', W']
            mask_down = (mask_down + 1) / 2  # [-1, 1] -> [0, 1]
            # Upsample to target size
            mask = F.interpolate(
                mask_down.unsqueeze(0),  # [1, 1, T', H', W']
                size=target_size,  # (T, H, W)
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)  # [1, T, H, W]
            mask = mask.clamp(0, 1)
        elif self.mask_mode == "downsample-project":
            # Use learned decoder: channel contraction + upsample in one step
            # Training: mask [0,1] -> *2-1 -> [-1,1] -> MaskEncoder -> mask_z [16,T',H',W']
            # Inference: mask_z -> MaskDecoder -> (+1)/2 -> [0,1] -> mask
            # Access underlying model for FSDP-wrapped models
            underlying_model = getattr(self.model, "module", self.model)
            mask_decoder = getattr(underlying_model, "mask_decoder", None)

            if mask_decoder is not None:
                # Ensure mask_decoder is on same device as input (handles offload_model case)
                input_device = mask_z.device
                if next(mask_decoder.parameters()).device != input_device:
                    mask_decoder.to(input_device)
                # Use learnable decoder (upsample + channel contraction)
                mask = mask_decoder(
                    mask_z.unsqueeze(0),  # [1, 16, T', H', W']
                    target_size,  # (T, H, W)
                ).squeeze(0)  # [1, T, H, W]
                mask = (mask + 1) / 2  # [-1, 1] -> [0, 1]
            else:
                # Fallback: average channels + trilinear interpolation
                logging.warning(
                    "mask_decoder not found, falling back to channel average + trilinear"
                )
                mask_down = mask_z.mean(
                    dim=0, keepdim=True
                )  # [16, T', H', W'] -> [1, T', H', W']
                mask_down = (mask_down + 1) / 2  # [-1, 1] -> [0, 1]
                mask = F.interpolate(
                    mask_down.unsqueeze(0),  # [1, 1, T', H', W']
                    size=target_size,  # (T, H, W)
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)  # [1, T, H, W]

            mask = mask.clamp(0, 1)
        elif self.mask_mode == "vae-project":
            # vae-project: decode through VAE (same as vae mode)
            # Note: project_out is applied during sampling, not here
            mask_3ch = self.vae.decode([mask_z])[0]  # [3, T, H, W]
            # Convert mask from [-1, 1] to [0, 1] and average channels
            mask = (mask_3ch.mean(dim=0, keepdim=True) + 1) / 2  # [1, T, H, W]
            mask = mask.clamp(0, 1)
        elif self.mask_mode == "vae-lora":
            if self.mask_vae_lora_decoder is None:
                raise RuntimeError("vae-lora mode requires mask_vae_lora_decoder")
            mask_3ch = self.mask_vae_lora_decoder.decode([mask_z])[0]
            mask = (mask_3ch.mean(dim=0, keepdim=True) + 1) / 2
            mask = mask.clamp(0, 1)
        elif self.mask_mode in ("mask-vae-project", "mask-vae-joint"):
            if self.mask_vae is None or self.mask_vae_proj_out is None:
                raise RuntimeError(
                    f"{self.mask_mode} mode requires mask_vae and mask_vae_proj_out"
                )
            mask_z_proj = self.mask_vae_proj_out(mask_z)
            # MaskVAE.decode expects 5D [B,C,T,H,W], but mask_z_proj is 4D [C,T,H,W]
            mask_z_proj_5d = mask_z_proj.unsqueeze(0)  # [1, C, T, H, W]
            mask_5d = self.mask_vae.decode(mask_z_proj_5d, target_shape=target_size)
            mask = mask_5d.squeeze(0)  # [1, T, H, W]
            mask = (mask + 1) / 2  # [-1, 1] -> [0, 1]
            mask = mask.clamp(0, 1)
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

        return {
            "full_video": full_video,  # [3, T, H, W] range [-1, 1]
            "background": background,  # [3, T, H, W] range [-1, 1]
            "foreground": foreground,  # [3, T, H, W] range [-1, 1]
            "mask": mask,  # [1, T, H, W] range [0, 1]
        }

    def save_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        output_dir: str,
        fps: int = 24,
        prefix: str = "output",
    ):
        """
        Save generated outputs to video files.

        Args:
            outputs: Dictionary with generated layers
            output_dir: Output directory
            fps: Output video FPS
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            import torchvision.io as tvio
        except ImportError:
            logging.warning("torchvision not available. Saving as tensors instead.")
            for name, tensor in outputs.items():
                torch.save(tensor, os.path.join(output_dir, f"{prefix}_{name}.pt"))
            return

        for name, tensor in outputs.items():
            # Move to CPU for video writing
            tensor = tensor.cpu()

            # Convert to uint8 video format
            if name == "mask":
                # Mask: [1, T, H, W] -> [T, H, W, 3]
                video = tensor.squeeze(0)  # [T, H, W]
                video = video.unsqueeze(-1).repeat(1, 1, 1, 3)  # [T, H, W, 3]
                video = (video * 255).clamp(0, 255).byte()
            else:
                # RGB: [3, T, H, W] -> [T, H, W, 3]
                video = tensor.permute(1, 2, 3, 0)  # [T, H, W, 3]
                video = ((video + 1) / 2 * 255).clamp(0, 255).byte()

            output_path = os.path.join(output_dir, f"{prefix}_{name}.mp4")
            tvio.write_video(output_path, video, fps=fps)
            logging.info(f"Saved: {output_path}")

        # Save foreground * mask composite (fg with alpha applied, black background)
        if "foreground" in outputs and "mask" in outputs:
            fg = outputs["foreground"].cpu()  # [3, T, H, W] range [-1, 1]
            mask = outputs["mask"].cpu()  # [1, T, H, W] range [0, 1]

            # Normalize fg to [0, 1] and apply mask
            fg_norm = (fg + 1) / 2  # [0, 1]
            fg_masked = fg_norm * mask  # [3, T, H, W] range [0, 1]

            # Convert to video format
            video = fg_masked.permute(1, 2, 3, 0)  # [T, H, W, 3]
            video = (video * 255).clamp(0, 255).byte()

            output_path = os.path.join(output_dir, f"{prefix}_fg_masked.mp4")
            tvio.write_video(output_path, video, fps=fps)
            logging.info(f"Saved: {output_path}")


def create_layered_pipeline(
    model_path: str,
    lora_path: Optional[str] = None,
    use_ema: bool = False,
    mask_mode: str = "vae",
    use_4d_rope: bool = True,
    rope_dim_ratios: Optional[Tuple[int, int, int, int]] = None,
    device_id: int = 0,
    model_size: str = "14B",
    mask_vae_path: Optional[str] = None,
    mask_vae_proj_path: Optional[str] = None,
    mask_vae_lora_path: Optional[str] = None,
) -> LayeredWanT2V:
    """
    Convenience function to create LayeredWanT2V pipeline.

    Args:
        model_path: Path to Wan2.1 checkpoint
        lora_path: Path to LoRA weights (optional)
        use_ema: Use EMA weights for inference (recommended for better quality)
        mask_mode: Mask processing mode ("vae", "downsample", "downsample-project", "vae-project", "vae-lora", "mask-vae-project", or "mask-vae-joint")
        use_4d_rope: Use 4D RoPE (L, T, H, W) instead of 3D (T, H, W)
        rope_dim_ratios: Custom 4D RoPE dimension allocation (L, T, H, W) in real dims
        device_id: GPU device ID
        model_size: Model size ("14B" or "1.3B")
        mask_vae_path: Path to MaskVAE checkpoint (mask-vae-project/mask-vae-joint mode)
        mask_vae_proj_path: Path to projection layers (vae-project/mask-vae-project/mask-vae-joint mode)
        mask_vae_lora_path: Path to VAE LoRA checkpoint (vae-lora mode)

    Returns:
        LayeredWanT2V pipeline instance
    """
    # Import config based on model size
    if model_size == "14B":
        from .configs.wan_t2v_14B import t2v_14B as config
    elif model_size == "1.3B":
        from .configs.wan_t2v_1_3B import t2v_1_3B as config
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    return LayeredWanT2V(
        config=config,
        checkpoint_dir=model_path,
        lora_path=lora_path,
        use_ema=use_ema,
        mask_mode=mask_mode,
        use_4d_rope=use_4d_rope,
        rope_dim_ratios=rope_dim_ratios,
        device_id=device_id,
        mask_vae_path=mask_vae_path,
        mask_vae_proj_path=mask_vae_proj_path,
        mask_vae_lora_path=mask_vae_lora_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layered Video Generation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights for inference (better quality)",
    )
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="vae",
        choices=[
            "vae",
            "downsample",
            "downsample-project",
            "vae-project",
            "vae-lora",
            "mask-vae-project",
            "mask-vae-joint",
        ],
        help="Mask processing mode: vae (encode through VAE), downsample (direct downsample + repeat channels), downsample-project (downsample + learnable projection), vae-project (VAE with learnable projection), vae-lora (VAE + project-in + decoder LoRA), mask-vae-project (dedicated MaskVAE with projection), mask-vae-joint (MaskVAE + projection joint training)",
    )
    parser.add_argument(
        "--mask_vae_path",
        type=str,
        default=None,
        help="Path to MaskVAE checkpoint (mask-vae-project/mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_proj_path",
        type=str,
        default=None,
        help="Path to mask projection checkpoint (vae-project/mask-vae-project/mask-vae-joint mode)",
    )
    parser.add_argument(
        "--mask_vae_lora_path",
        type=str,
        default=None,
        help="Path to VAE LoRA checkpoint (vae-lora mode)",
    )
    # 4D RoPE arguments
    parser.add_argument(
        "--use_4d_rope",
        dest="use_4d_rope",
        action="store_true",
        help="Use 4D RoPE (L, T, H, W) position encoding (default)",
    )
    parser.add_argument(
        "--no_4d_rope",
        dest="use_4d_rope",
        action="store_false",
        help="Use original 3D RoPE (T, H, W) position encoding",
    )
    parser.set_defaults(use_4d_rope=True)
    parser.add_argument(
        "--rope_dim_ratios",
        type=str,
        default=None,
        help="4D RoPE dimension allocation (L,T,H,W) as comma-separated ints, e.g. '8,42,40,38'",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--fg_prompt", type=str, default="")
    parser.add_argument("--bg_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    # Parse rope_dim_ratios
    rope_dim_ratios = None
    if args.rope_dim_ratios:
        rope_dim_ratios = tuple(int(x) for x in args.rope_dim_ratios.split(","))
        if len(rope_dim_ratios) != 4:
            raise ValueError(
                f"rope_dim_ratios must have 4 values (L,T,H,W), got {len(rope_dim_ratios)}"
            )

    # Create pipeline
    pipeline = create_layered_pipeline(
        model_path=args.model_path,
        lora_path=args.lora_path,
        use_ema=args.use_ema,
        mask_mode=args.mask_mode,
        use_4d_rope=args.use_4d_rope,
        rope_dim_ratios=rope_dim_ratios,
        mask_vae_path=args.mask_vae_path,
        mask_vae_proj_path=args.mask_vae_proj_path,
        mask_vae_lora_path=args.mask_vae_lora_path,
        device_id=args.device,
    )

    # Generate
    outputs = pipeline.generate(
        input_prompt=args.prompt,
        fg_prompt=args.fg_prompt,
        bg_prompt=args.bg_prompt,
        size=(args.width, args.height),
        frame_num=args.frames,
        sampling_steps=args.steps,
        seed=args.seed,
    )

    # Save
    pipeline.save_outputs(outputs, args.output_dir, args.fps)
    print(f"Generated outputs saved to {args.output_dir}")
