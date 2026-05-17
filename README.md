# LayerT2V: A Unified Multi-Layer Video Generation Framework

<div align="center">

**Guangzhao Li, Kangrui Cen, Baixuan Zhao, Yi Xin, Siqi Luo, Guangtao Zhai, Lei Zhang, Xiaohong Liu**

[![arXiv](https://img.shields.io/badge/arXiv-2508.04228-b31b1b.svg)](https://arxiv.org/abs/2508.04228)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://layert2v.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Accepted-brightgreen.svg)](https://arxiv.org/abs/2508.04228)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-orange.svg)](#release-status)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Coming%20Soon-orange.svg)](#release-status)
[![Dataset](https://img.shields.io/badge/Dataset-Coming%20Soon-orange.svg)](#release-status)

</div>

This is the official repository for **LayerT2V: A Unified Multi-Layer Video Generation Framework**.

Our paper has been accepted. We are preparing the public release of the code, pretrained checkpoints, and dataset resources.

## News

- **[2026.05]** Our paper has been accepted. Code, checkpoints, and dataset resources are being prepared for release.
- **[2026.02]** We updated the arXiv version of LayerT2V.
- **[2025.08]** We released the LayerT2V preprint on arXiv.

## Overview

**LayerT2V** is a unified multi-layer video generation framework for layer-aware text-to-video synthesis. Instead of generating only a final composited video, LayerT2V produces an editable layered representation in a single inference pass, including:

- the full composited video,
- an independent background layer,
- multiple foreground RGB layers,
- corresponding foreground alpha mattes.

The framework serializes multiple layer representations along the temporal dimension and models them jointly with a shared video generation trajectory. This design improves semantic alignment, temporal coherence, and cross-layer consistency while keeping the generation process unified.

LayerT2V further introduces layer-aware conditioning modules, including **LayerAdaLN** and layer-aware cross-attention modulation, to reduce layer ambiguity and conditional leakage. The training pipeline contains three stages: alpha mask VAE adaptation, joint multi-layer learning, and multi-foreground extension.

We also introduce **VidLayer**, a large-scale dataset for multi-layer video generation.

## Release Status

| Component | Status |
| --- | --- |
| Paper | Accepted |
| Project page | Available |
| Source code | Coming soon |
| Pretrained checkpoints | Coming soon |
| VidLayer dataset | Coming soon |
| Inference scripts | Coming soon |
| Training scripts | Coming soon |
| Evaluation scripts | Coming soon |

We are cleaning the implementation, organizing model weights, and preparing dataset release materials. Please watch this repository for updates.

## Installation

Installation instructions will be released together with the source code.

```bash
git clone https://github.com/guangzhaoli/LayerT2V.git
cd LayerT2V
```

The public release will include environment files, dependency versions, and setup instructions.

## Inference

Inference code and pretrained checkpoints will be released soon.

The release will include:

- checkpoint download instructions,
- text-to-video generation examples,
- multi-layer generation examples,
- alpha matte and foreground extraction examples,
- visualization utilities for generated layers.

## Dataset

The VidLayer dataset will be released after final organization and license checks.

The dataset release will include:

- download instructions,
- data format documentation,
- preprocessing scripts,
- training and evaluation splits.

## Training

Training scripts and configuration files will be released with the source code.

The release will include scripts for the main training stages:

- alpha mask VAE adaptation,
- joint multi-layer learning,
- multi-foreground extension.

## Evaluation

Evaluation scripts will be released with the source code.

The release will include commands and instructions for evaluating:

- visual fidelity,
- temporal consistency,
- cross-layer coherence,
- alpha matte quality,
- foreground-background consistency.

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

## Acknowledgments

We thank everyone who supports this project. More details will be added together with the full code release.

## Contact

For questions, please open an issue in this repository.
