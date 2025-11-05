<div align="center">

<img src="VLAb.png" alt="Logo" width="300">

# Your Laboratory for Pretraining VLAs 

A streamlined library for pretraining VLA models, derived from LeRobot, used to pretrain SmolVLA but focused specifically on pretraining workflows. This library enables efficient pretraining of vision-language-action models on a cluster, multi-gpu setups and on multiple datasets for robotics applications.

</div>

## Table of Contents

- [Features](#features)
- [Who is this Library for?](#who-is-this-library-for)
- [Installation](#installation)
  - [Verify Installation](#verify-installation)
  - [Configure HuggingFace Hub](#configure-huggingface-hub-if-using-remote-datasets)
- [Get the data](#get-the-data)
- [Usage](#usage)
  - [Training with Accelerate](#training-with-accelerate)
  - [SLURM Cluster Training](#slurm-cluster-training)
- [Additional Resources](#additional-resources)
- [Citation](#citation)
- [File Structure](#file-structure)

> Note on video backends

> By default we avoid pinning system FFmpeg in the environment to prevent compatibility issues with TorchCodec on some systems. The training pipeline works with alternative video backends (PyAV, OpenCV, ImageIO). If you specifically need TorchCodec, install FFmpeg in your environment and ensure the FFmpeg libraries match TorchCodec's compatibility table. Otherwise, set `--dataset.video_backend=pyav` (default) or switch to OpenCV/ImageIO in your data loader.

## Features

It is directly compatible with the https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v1 and https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2 that are used to pretrain smolvla. Note that, while this library can be efficient for pretraining, for finetuning we recommend using lerobot as it is more up-to-date with the inference pipeline and new hardware. 

## Who is this Library for?

**Pretraining-Focused**: Streamlined codebase designed specifically for VLA pretraining workflows—environment simulation and evaluation dependencies removed to enable cleaner pretraining on real-world data pipelines
**Simple Setup**: Get started immediately with single-command environment creation using conda env create -f environment.yml
**Distributed Training**: Seamless multi-GPU and multi-node support through Accelerate, tested on both single machines and SLURM clusters
**Multi-Dataset Training:** Easily train on multiple datasets with configurable sampling strategies and automatic data loading
**Reduced Dependencies:** Leaner footprint compared to full robotics frameworks in a favor of faster installation and fewer potential conflicts.

> **Note on Fine-tuning & Inference**: Pretrained policies from this codebase may not be directly compatible with the latest LeRobot version. To fine-tune or run inference using LeRobot, you'll need to convert your checkpoint using the [migration script](https://github.com/huggingface/lerobot/blob/f6b16f6d97155e3ce34ab2a1ec145e9413588197/src/lerobot/processor/migrate_policy_normalization.py#L4) to ensure compatibility with updated normalization formats.

## Installation

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate vlab
```

**Note:** If your system uses CUDA 12.1, edit `environment.yml` and set `pytorch-cuda=12.1` before creating the environment.

Ensure your Python path includes the `src` directory (e.g., set `PYTHONPATH` to include `src`).

### Verify Installation

Run the test script to verify everything is set up correctly:

```bash
python test_installation.py
```

This will output:
```
============================================================
VLAb Installation Test
============================================================
✓ TrainPipelineConfig
✓ Policy factory
✓ Dataset factory
============================================================
✅ All tests passed!
```

### Configure HuggingFace Hub (if using remote datasets)

```bash
# Login to HuggingFace (if you need to download/upload models/datasets)
huggingface-cli login
```
## Get the data

This README uses SmolVLA pretraining datasets as examples, but VLAb supports any LeRobot-format datasets from Hugging Face Hub. Pass multiple datasets as comma-separated repository IDs (e.g., `username/dataset1,username/dataset2`).

VLAb works directly with SmolVLA community datasets that follow the LeRobot format:

- **[Community Dataset v1](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v1)**: 128 datasets from 55 contributors (11K episodes, 46.9 hours)
- **[Community Dataset v2](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2)**: Updated collection with v2.0/v2.1 format support

For dataset details, statistics, and structure, see the dataset cards!

```bash
# Download v2.1 (recommended)
huggingface-cli download HuggingFaceVLA/community_dataset_v1 \
  --local-dir /path/to/local/datasets/community_dataset_v1

# Download older v2.0
huggingface-cli download HuggingFaceVLA/community_dataset_v2 \
  --local-dir /path/to/local/datasets/community_dataset_v2
```

To train from the Hub without pre-downloading, pass just the repo id.

## Usage

### Training with Accelerate

Accelerate works for both single-GPU and multi-GPU setups. First, configure accelerate:

```bash
accelerate config
```

Train on datasets from the Hugging Face Hub (automatically downloaded):

```bash
accelerate launch src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="HuggingFaceVLA/community_dataset_v1,HuggingFaceVLA/community_dataset_v2" \
    --dataset.video_backend=pyav \
    --output_dir="./outputs/training" \
    --batch_size=8 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"
```

To train on local datasets, specify the root directory with `--dataset.root`:

```bash
accelerate launch src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="user1/dataset_a,user2/dataset_b" \
    --dataset.root="/path/to/local/datasets" \
    --dataset.video_backend=pyav \
    --output_dir="./outputs/training" \
    --batch_size=8 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"
```

### SLURM Cluster Training

For SLURM clusters, we provide two sample scripts to help you get started:

Fresh training: Use this script to start training from scratch
Resume training: Use this script to continue training from a checkpoint

Edit the scripts according to your cluster configuration, then submit:
```bash
sbatch scripts/training/train_smolvla_optimized_fresh.slurm
sbatch scripts/training/train_smolvla_resume.slurm
```

## Additional Resources

- **[LeRobot GitHub](https://github.com/huggingface/lerobot)**: Main LeRobot repository
- **[Finetune SmolVLA with lerobot](https://huggingface.co/docs/lerobot/smolvla)**: Complete guide for fine-tuning the checkpoint using LeRobot
- **[LeRobot Installation Guide](https://huggingface.co/docs/lerobot/en/installation)** 
- **[Accelerate Documentation](https://huggingface.co/docs/accelerate)**:

## Citation

If you use this library or models trained with it in your research, please cite the codebase:

```bibtex
@misc{aubakirova2025vlab,
  author = {Dana Aubakirova and Mustafa Shukor and Jade Cholgari and Leandro von Werra},
  title = {VLAb: Your Laboratory for Pretraining VLAs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/vlab}}
}

And the paper:

```bibtex
@article{shukor2025smolvla,
  title   = {SmolVLA: A vision-language-action model for affordable and efficient robotics},
  author  = {Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  year    = {2025},
  journal = {arXiv preprint},
  eprint  = {2506.01844},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

## File Structure

```
VLAb/
├── src/lerobot/                  # Source code
│   ├── configs/                  # Configuration classes
│   ├── datasets/                 # Dataset handling
│   ├── optim/                    # Optimizers and schedulers
│   ├── policies/                 # Policy implementations
│   │   └── smolvla2/            # SmolVLA2 specific code
│   ├── scripts/                  # Training scripts
│   └── utils/                    # Utility functions
├── scripts/training/             # SLURM training scripts
├── examples/                     # Example scripts
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

