# VLAb Examples

This directory contains example scripts demonstrating how to use VLAb on the cluster.

For general usage and installation instructions, see the main [README.md](../README.md).

## Structure

- `scripts/` - Example training scripts (SLURM scripts for cluster training)
- `all_datasets_relative/` - Paths to the pretraining datasets  from Community_Datasets_v1, and Community_Satasets_v2 that are stored locally

## Training Scripts

The `scripts/` directory contains SLURM scripts for multi-GPU training on compute clusters:

### Fresh Training (`train_smolvla_optimized_fresh.slurm`)

This script starts training from scratch with a new output directory. It uses all datasets from `all_datasets_relative.txt` and configures an 8-GPU training setup.

**Usage:**
```bash
sbatch examples/scripts/train_smolvla_optimized_fresh.slurm
```

### Resume Training (`train_smolvla_resume.slurm`)

This script resumes training from a previous checkpoint. It loads the configuration from the checkpoint and continues training.


**Usage:**
```bash
# First, set the checkpoint path in the script:
export RESUME_FROM_CHECKPOINT="/path/to/your/checkpoint"

# Then submit:
sbatch examples/scripts/train_smolvla_resume.slurm
```

## Dataset List Generation

You can generate a dataset list from local lerobot datasets using the following script:

```bash
# Generate dataset list dynamically
DATASET_LIST=$(python -c "
import os
datasets = []
for root, dirs, files in os.walk('/path/to/local/datasets'):
    if 'data.parquet' in files:
        rel_path = os.path.relpath(root, '/path/to/local/datasets')
        datasets.append(rel_path.replace('/', '/'))
print(','.join(datasets))
")

# Train with generated list
accelerate launch --config_file accelerate_configs/multi_gpu.yaml \
    src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="$DATASET_LIST" \
    --dataset.root="/path/to/local/datasets" \
    --policy.repo_id="/path/to/the/hub_username/" \
    --dataset.video_backend=pyav \
    --output_dir="./outputs/multi_gpu_training" \
    --batch_size=8 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"
```

**Note:** Update `/path/to/local/datasets` to point to your actual dataset directory. This script will automatically discover all datasets that contain a `data.parquet` file.

## Quick Start

For basic usage, see the main [README.md](../README.md).

For cluster training, use the SLURM scripts in this directory. Make sure to:
1. Update all paths to match your environment
2. Adjust resource requests (GPUs, CPUs, memory, time)
3. Configure your dataset paths and list whether you are using the datasets on the hub or locally
4. Set up your WandB credentials if using logging
