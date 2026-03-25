<!-- # Official Implementation of MambaEye -->


<div align="center">
<h1>MambaEye </h1>
<h3>A Size-Agnostic Visual Encoder with Causal Sequential Processing</h3>

[Changho Choi](https://usingcolor.github.io/)<sup>1</sup> ,[Minho Kim](https://scholar.google.com/citations?user=NU-Km1QAAAAJ&hl=en)<sup>2</sup> ,[Jinkyu Kim](https://visionai.korea.ac.kr/team/)<sup>1&dagger;</sup>

<sup>1</sup> Korea University
<sup>2</sup> MIT

<sup>&dagger;</sup> Corresponding author.

</div>


[![Venue](https://img.shields.io/badge/Venue-CVPR_2026_Findings-blue)](https://cvpr.thecvf.com/2026)
[![HuggingFace Checkpoints](https://img.shields.io/badge/HuggingFace-Checkpoints-blue?logo=huggingface&style=flat-square)](https://huggingface.co/collections/usingcolor/mambaeye)
[![arXiv](https://img.shields.io/badge/arXiv-2511.19963-red)](https://arxiv.org/abs/2511.19963)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://usingcolor.github.io/MambaEye/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)


<video src="https://github.com/usingcolor/MambaEye/blob/main/assets/inference.mp4" controls autoplay muted playsinline></video>
<video src="https://github.com/usingcolor/MambaEye/blob/main/assets/onestep_animation.mp4" controls autoplay muted playsinline></video>

## Overview

This repository contains the official PyTorch implementation for training and inference of *MambaEye*.

Key features of the architecture include:
- **Flexible Image Understanding:** Processes multi-resolution images and arbitrary aspect ratios, including partial images.
- **Variable-Length Processing:** Natively handles sequential inputs of varying lengths.
- **Efficient Scaling:** Achieves linear memory and computational complexity (by number of patches) powered by Mamba2 layers. (Constant memory for inference.)

## TODO
- [ ] Update for Camera-Ready version.
- [ ] Write a blog post (if #of stars > 100)

## Installation

Ensure you have Python 3.12+ installed and a CUDA-capable environment. We strongly recommend creating an isolated environment:

```bash
# Environment setup (first time only)
source scripts/env_setup.sh

# Activate the environment (after setup)
source .venv/bin/activate
```

*(Note: `mamba-ssm==2.2.4` and `causal-conv1d` may require specific CUDA versions. Check the [official Mamba repo](https://github.com/state-spaces/mamba) if you face compilation issues. We tested this environment with RTX 4090, L40S, A100, H100, H200 GPUs.)*

Test environment with python script:
```python
import torch
from mamba_ssm import Mamba2
model=Mamba2(256, 4, 2, 1).cuda()
x = torch.randn(1, 10, 256).cuda()
y = model(x)
```


## Inference

### Model Weights
All the model weights are uploaded in HuggingFace. You can download them from [here](https://huggingface.co/collections/usingcolor/mambaeye).

<!-- table for weights -->

| Model | Params(M) | Trained Sequence Length | Top-1 Accuracy (512x512) | Link |
| --- | --- | --- | --- | --- |
| MambaEye-Tiny | 5.8 | 1024 | 66.2% | [Link](https://huggingface.co/usingcolor/MambaEye-tiny/resolve/main/mambaeye_tiny.pt) |
| MambaEye-Tiny (FT) |5.8| 2048 | 67.2% | [Link](https://huggingface.co/usingcolor/MambaEye-tiny/resolve/main/mambaeye_tiny_ft.pt) |
| MambaEye-Small | 11.0| 1024 | 72.7% | [Link](https://huggingface.co/usingcolor/MambaEye-small/resolve/main/mambaeye_small.pt) |
| MambaEye-Small (FT) |11.0| 2048 | 73.1% | [Link](https://huggingface.co/usingcolor/MambaEye-small/resolve/main/mambaeye_small_ft.pt) |
| MambaEye-Base |21.3| 1024 | 73.5% | [Link](https://huggingface.co/usingcolor/MambaEye-base/resolve/main/mambaeye_base.pt) |
| MambaEye-Base (FT) |21.3| 2048 | 75.0% | [Link](https://huggingface.co/usingcolor/MambaEye-base/resolve/main/mambaeye_base_ft.pt) |

### Inference Command
To evaluate our model at different sequence lengths and resolutions (as reported in the paper), you can use the provided inference scripts:

#### Single Image
For a single image, you can use the following command:

```bash
python eval.py \
    image_path=/path/to/image.jpg \
    ckpt_path=/path/to/checkpoint.ckpt \
    scan_pattern=random \
    resize_mode=none

# Use official model weights
python eval.py \
    image_path=/path/to/image.jpg \
    model_name=small-ft
```

#### ImageNet Validation Set
Download ImageNet val dataset and organize it in the standard PyTorch format as [Training section](#data-preparation).

```bash
# Using a local checkpoint
python eval.py \
    dataset.val.img_dir=/path/to/val \
    ckpt_path=/path/to/checkpoint.ckpt \
    scan_pattern=random \
    resize_mode=none

# Or automatically download and use an official model by its alias
# (Options: tiny, tiny-ft, small, small-ft, base, base-ft)
python eval.py \
    dataset.val.img_dir=/path/to/val \
    model_name=base-ft
```


## Training

### Data Preparation

Organize your [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset in the standard PyTorch format:
```text
data/imagenet/
  train/
    n01440764/
      n01440764_10026.JPEG
      ...
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
      ...
```


### Training Command
We use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training. All model layers and data settings are configured using [Hydra](https://hydra.cc/docs/intro/), with YAML files located in `configs/`.

> [!IMPORTANT]
> **Before training**, please ensure your environment configurations are set appropriately. You can either modify the YAML files directly or override them via the command line:
> - **Dataset Path:** Edit [`configs/dataset/default.yaml`](configs/dataset/default.yaml) and set `dataset.train.img_dir` and `dataset.val.img_dir` to point to your local ImageNet directories.
> - **GPU Settings:** Adjust [`configs/trainer/default.yaml`](configs/trainer/default.yaml) and set `dataloader.train.batch_size` and `trainer.accumulate_grad_batches` depending on your GPU memory.
> - **W&B Logging:** Add [`wandb.entity=YOUR_ENTITY wandb.project=YOUR_PROJECT`](configs/config.yaml) to enable Weights & Biases logging (otherwise it will fall back to CSV logging).

```bash
# Example: Training a 48-layer base model on ImageNet
python train.py model=base_48layers
```

To finetune an existing checkpoint:
```bash
python train.py \
  model=base_48layers \
  fine_tuning=true \
  ckpt_path=/path/to/checkpoint.ckpt # or mambaeye_small.pt
```


## Project Structure

```text
MambaEye/
├── assets/                       # Asset files
├── configs/                      # YAML configuration files
├── mambaeye/                     # Core module
│   ├── __init__.py
│   ├── dataset.py                # Dataset loading for ImageNet
│   ├── loss.py                   # Custom loss functions
│   ├── mambaeye_pl.py            # PyTorch Lightning module definitions
│   ├── model.py                  # Core MambaEye SSM architecture
│   ├── positional_encoding.py    # Positional encoding
│   └── scan.py                   # Scan pattern generation
├── scripts/                      # Utility scripts
├── eval.py                       # Standard inference script
├── train.py                      # Training script for MambaEye
├── requirements.txt              # Dependency requirements
└── README.md                     # This file
```




## Citation

If you find this code or our paper useful for your research, please cite our CVPR 2026 Findings track paper:

```bibtex
@inproceedings{mambaeye2026,
  title={MambaEye: A Size-Agnostic Visual Encoder with Causal Sequential Processing},
  author={Changho Choi and Minho Kim and Jinkyu Kim},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026},
  note={Accepted}
}
```


## Reference

- [Official Mamba Implementation](https://github.com/state-spaces/mamba)

## License

[MIT License](LICENSE)