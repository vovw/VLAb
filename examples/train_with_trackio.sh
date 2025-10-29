#!/bin/bash
# Example script showing how to use TrackIO instead of WandB for SmolVLA2 training

# Set up environment
export PYTHONPATH="/fsx/dana_aubakirova/vla/VLAb/src:$PYTHONPATH"

# TrackIO configuration
export TRACKIO_PROJECT="smolvla2-training"
export TRACKIO_NOTES="SmolVLA2 training with TrackIO - local experiment tracking"

# Optional: Host dashboard on HuggingFace Spaces
# export TRACKIO_SPACE_ID="your-username/your-space-name"

# Training configuration
export OUTPUT_DIR="./outputs/trackio_training_$(date +%Y%m%d_%H%M%S)"
export REPO_IDS="AndrejOrsula/lerobot_double_ball_stacking_random"
export VLM_REPO_ID="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

echo "🚀 Starting SmolVLA2 training with TrackIO logging"
echo "📊 Project: $TRACKIO_PROJECT"
echo "📁 Output: $OUTPUT_DIR"

# Run training with TrackIO enabled
python VLAb/src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="$REPO_IDS" \
    --dataset.root="/fsx/dana_aubakirova/vla/community_dataset_v1" \
    --policy.repo_id=$VLM_REPO_ID \
    --output_dir=$OUTPUT_DIR \
    --batch_size=4 \
    --steps=1000 \
    --eval_freq=-1 \
    --save_freq=500 \
    --wandb.enable=false \
    --trackio.enable=true \
    --trackio.project="$TRACKIO_PROJECT" \
    --trackio.notes="$TRACKIO_NOTES" \
    $([ -n "$TRACKIO_SPACE_ID" ] && echo "--trackio.space_id=$TRACKIO_SPACE_ID" || echo "")

echo "✅ Training completed!"
echo "📊 To view results, run: trackio show"
