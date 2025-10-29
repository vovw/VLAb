# SmolVLA2 Pretraining Library

A streamlined library for pretraining SmolVLA2 models, derived from LeRobot but focused specifically on pretraining workflows.

## Features

- 🚀 **Optimized for Pretraining**: Removed environment/evaluation dependencies for faster, cleaner pretraining
- 📊 **Dual Logging Support**: Choose between WandB or TrackIO for experiment tracking
- 🔧 **Multi-GPU Training**: Optimized for distributed training with Accelerate
- 🎯 **SmolVLA2 Focus**: Specialized for SmolVLA2 model architecture
- 🧹 **Clean Codebase**: Removed unused components for better maintainability

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch transformers accelerate datasets
pip install wandb  # For WandB logging (optional)
pip install trackio  # For TrackIO logging (optional)

# Set up Python path
export PYTHONPATH="/path/to/smolvla2_pretraining/src:$PYTHONPATH"
```

### Basic Training

```bash
# Train with WandB logging
python src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="your-dataset" \
    --output_dir="./outputs/training" \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"

# Train with TrackIO logging
python src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="your-dataset" \
    --output_dir="./outputs/training" \
    --trackio.enable=true \
    --trackio.project="smolvla2-training"
```

### SLURM Training

Use the optimized SLURM script for multi-GPU training:

```bash
# Edit the script to set your preferences
vim scripts/training/train_smolvla_optimized_fresh.slurm

# Change logging backend in the script:
export LOGGING_BACKEND="trackio"  # or "wandb" or "local"

# Submit job
sbatch scripts/training/train_smolvla_optimized_fresh.slurm
```

## Logging Options

### WandB (Weights & Biases)
```bash
--wandb.enable=true \
--wandb.project="your-project" \
--wandb.entity="your-entity" \
--wandb.notes="Training notes"
```

### TrackIO (Local-first)
```bash
--trackio.enable=true \
--trackio.project="your-project" \
--trackio.notes="Training notes" \
--trackio.space_id="username/space-name"  # Optional: HF Space hosting
```

### Local Only
```bash
--wandb.enable=false \
--trackio.enable=false
```

## TrackIO Integration

TrackIO provides a local-first alternative to WandB with the following benefits:

- 📱 **Local Dashboard**: View experiments locally with `trackio show`
- 🌐 **Optional Cloud Hosting**: Host on HuggingFace Spaces
- 🔄 **WandB Compatible API**: Drop-in replacement for WandB
- 💾 **Local Storage**: All data stored locally by default

### TrackIO Setup

1. Install TrackIO:
   ```bash
   pip install trackio
   ```

2. Run training with TrackIO:
   ```bash
   ./examples/train_with_trackio.sh
   ```

3. View results:
   ```bash
   trackio show
   ```

## Configuration

### Model Configuration
- **Policy Type**: `smolvla2`
- **VLM Model**: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- **PEFT Method**: LoRA with configurable rank
- **Attention**: Cross-attention with causal masking

### Training Configuration
- **Batch Size**: Configurable per GPU (default: 8)
- **Learning Rate**: Separate rates for VLM and policy
- **Scheduler**: Cosine decay with warmup
- **Mixed Precision**: Configurable AMP support

### Dataset Configuration
- **Multi-dataset**: Support for multiple datasets
- **Image Processing**: Configurable transforms and dimensions
- **Video Backend**: PyAV for video processing
- **FPS Control**: Min/max FPS filtering

## Cleanup

The library includes a cleanup script to remove unused files:

```bash
# Dry run (preview what will be removed)
python cleanup_unused_files.py

# Actually remove files
python cleanup_unused_files.py --live
```

### Removed Components
- Robot control utilities
- Environment simulation code
- Online RL buffers
- Dataset conversion utilities (v2/v21)
- Visualization utilities (Rerun)
- Benchmarking utilities

## Examples

### Multi-GPU Training
```bash
# 4-GPU training with TrackIO
export LOGGING_BACKEND="trackio"
sbatch scripts/training/train_smolvla_optimized_fresh.slurm
```

### Custom Dataset
```bash
python src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="your-username/your-dataset" \
    --dataset.root="/path/to/local/datasets" \
    --output_dir="./outputs/custom_training" \
    --batch_size=4 \
    --steps=50000 \
    --trackio.enable=true
```

### Resume Training
```bash
python src/lerobot/scripts/train.py \
    --config_path="./outputs/previous_training/train_config.json" \
    --resume=true
```

## File Structure

```
smolvla2_pretraining/
├── src/lerobot/
│   ├── configs/          # Configuration classes
│   ├── datasets/         # Dataset handling (core only)
│   ├── optim/           # Optimizers and schedulers
│   ├── policies/        # SmolVLA2 policy implementation
│   ├── scripts/         # Training scripts
│   └── utils/           # Core utilities only
├── scripts/training/    # SLURM training scripts
├── examples/           # Example usage scripts
├── cleanup_unused_files.py  # Cleanup script
└── README.md           # This file
```

## Contributing

When adding new features:
1. Keep the pretraining focus - avoid environment/evaluation code
2. Maintain compatibility with both WandB and TrackIO
3. Update configuration classes as needed
4. Add examples for new features

## License

Apache 2.0 License (inherited from LeRobot)
