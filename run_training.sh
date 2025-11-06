#!/bin/bash
# Simple script to run training without accelerate

set +u
source /fsx/dana_aubakirova/miniconda/etc/profile.d/conda.sh
conda activate vlab

# Set PYTHONPATH
export PYTHONPATH="/fsx/dana_aubakirova/vla/VLAb/src:${PYTHONPATH:-}"

# Change to VLAb directory
cd /fsx/dana_aubakirova/vla/VLAb

# Run training with unbuffered output
python -u src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="Beegbrain/pick_lemon_and_drop_in_bowl,Chojins/chess_game_000_white_red" \
    --dataset.video_backend=pyav \
    --output_dir="./outputs/training" \
    --batch_size=8 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"

