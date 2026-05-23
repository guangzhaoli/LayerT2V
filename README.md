<p align="center">
  <h1 align="center">LayerT2V: A Unified Multi-Layer Video Generation Framework</h1>
</p>

<p align="center">
  <strong>Guangzhao Li<sup>*</sup></strong>
  ·
  <strong>Kangrui Cen<sup>*</sup></strong>
  ·
  <strong>Baixuan Zhao</strong>
  ·
  <strong>Yi Xin</strong>
  ·
  <strong>Siqi Luo</strong>
  ·
  <strong>Guangtao Zhai</strong>
  ·
  <strong>Lei Zhang</strong>
  ·
  <strong>Xiaohong Liu</strong>
  <br>
  <br>
  <a href="https://arxiv.org/abs/2508.04228"><img src="https://img.shields.io/badge/arXiv-2508.04228-b31b1b.svg" alt="arXiv"></a>
  <a href="https://layert2v.github.io/"><img src="https://img.shields.io/badge/Project_Page-LayerT2V-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2508.04228"><img src="https://img.shields.io/badge/Paper-ICML-brightgreen" alt="Paper"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License"></a>
</p>

<p align="center">
  <img src="assets/pipe.png" width="100%" alt="LayerT2V pipeline">
  <br>
  <em>LayerT2V jointly generates a composited video, background layer, foreground RGB layers, and alpha mattes in a unified video generation trajectory.</em>
</p>

---

This is the official repository for **LayerT2V: A Unified Multi-Layer Video Generation Framework**.

## Abstract

**TL;DR:** LayerT2V is a unified layer-aware text-to-video generation framework that produces editable multi-layer video representations in a single inference pass.

<details>
<summary>Click to read the full abstract</summary>

LayerT2V targets layer-aware video generation, where the model generates not only the final composited video but also an independent background layer, multiple foreground RGB layers, and corresponding alpha mattes. It serializes multiple layer representations along the temporal dimension and jointly models them on a shared generation trajectory, improving semantic alignment, temporal coherence, and cross-layer consistency.

To reduce layer ambiguity and conditional leakage, LayerT2V introduces layer-aware conditioning modules, including **LayerAdaLN** and layer-aware cross-attention modulation. The training pipeline contains alpha mask VAE adaptation, joint multi-layer learning, and multi-foreground extension. We also introduce **VidLayer**, a large-scale dataset for multi-layer video generation.

</details>

---

## Key Features

- **Unified multi-layer generation:** Produces the full video, background, foreground RGB layers, and alpha mattes in one inference pass.
- **Layer-aware diffusion modeling:** Serializes layer representations along the temporal dimension and models them jointly.
- **Layer-specific conditioning:** Uses LayerAdaLN and layer-aware cross-attention modulation to reduce layer ambiguity.
- **Editable video representation:** Generates explicit foreground/background layers for downstream editing and compositing.
- **VidLayer dataset:** Introduces a large-scale dataset designed for multi-layer video generation.

---

## News

- **[2026.05]** LayerT2V is accepted to ICML.
- **[2026.02]** We updated the arXiv version of LayerT2V.
- **[2025.08]** We released the LayerT2V preprint on arXiv.

---

## Todo

- [x] Release the initial codebase.
- [x] Release the project page.
- [x] Release the arXiv paper.
- [ ] Release pretrained LayerT2V checkpoints.
- [ ] Release VidLayer dataset resources.
- [ ] Add detailed evaluation scripts and instructions.

---

## Getting Started

### Installation

```bash
git clone https://github.com/guangzhaoli/LayerT2V.git
cd LayerT2V

conda create -n layert2v python=3.10
conda activate layert2v

pip install -r requirements.txt
```

For training, install the training dependencies as needed:

```bash
pip install -r requirements_training.txt
```

### Model Preparation

LayerT2V builds on Wan2.1 video generation components. Prepare the required Wan2.1 base checkpoint locally and pass its path through `--model_path`.

Pretrained LayerT2V checkpoints and VidLayer dataset release links will be added after release preparation is complete.

---

## How to Use

### Inference

Use the provided inference script for layered video generation:

```bash
bash scripts/run_inference.sh \
  --model_path /path/to/Wan2.1-T2V-1.3B \
  --lora_path /path/to/layert2v/checkpoint \
  --prompt "A ship sails on the ocean under blue sky." \
  --fg_prompt "A ship." \
  --bg_prompt "A vast ocean under blue sky." \
  --output_dir outputs/demo
```

Useful options include:

- `--mask_mode`: mask processing mode, such as `vae`, `downsample`, `vae-project`, `mask-vae-project`, or `mask-vae-joint`.
- `--mask_vae_path`: MaskVAE checkpoint path for MaskVAE-based modes.
- `--mask_vae_proj_path`: projection checkpoint path for projection-based modes.
- `--use_4d_rope` / `--no_4d_rope`: switch between 4D layer-temporal-spatial RoPE and original 3D RoPE.
- `--width`, `--height`, `--frames`, `--steps`, `--seed`: generation controls.

### Training

Use the training script with a config file, Wan2.1 checkpoint, and dataset root:

```bash
bash scripts/run_train.sh \
  --config training/configs/train.yaml \
  --num_gpus 4 \
  --model_path /path/to/Wan2.1-T2V-1.3B \
  --data_root /path/to/VidLayer \
  --output_dir outputs/train
```

The training code supports the main LayerT2V stages, including alpha mask VAE adaptation, joint multi-layer learning, and multi-foreground extension.

---

## Repository Structure

```text
LayerT2V
├── generate.py
├── inference_res.py
├── scripts/
├── training/
├── vis_scripts/
└── wan/
```

- `wan/`: Wan2.1-based generation modules and LayerT2V inference implementation.
- `training/`: training datasets, configs, and training entrypoints.
- `scripts/`: launch scripts for inference, training, batch inference, and experiment utilities.
- `vis_scripts/`: visualization utilities for generated layer outputs.

---

## Citation

If you find LayerT2V useful for your research, please cite our paper:

```bibtex
@misc{li2025layert2v,
  title = {LayerT2V: A Unified Multi-Layer Video Generation Framework},
  author = {Li, Guangzhao and Cen, Kangrui and Zhao, Baixuan and Xin, Yi and Luo, Siqi and Zhai, Guangtao and Zhang, Lei and Liu, Xiaohong},
  year = {2025},
  eprint = {2508.04228},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2508.04228},
  url = {https://arxiv.org/abs/2508.04228}
}
```

The citation will be updated after the official proceedings information becomes available.

---

## Acknowledgments

We thank everyone who supports this project. More details will be added together with future releases.

## Contact

For questions, please open an issue in this repository.
