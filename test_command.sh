#!/bin/bash
# Test script to validate the training command

set +u  # Disable unbound variable checking for conda activation
source /fsx/dana_aubakirova/miniconda/etc/profile.d/conda.sh
conda activate vlab

# Set PYTHONPATH (handle case where it might not be set)
export PYTHONPATH="/fsx/dana_aubakirova/vla/VLAb/src:${PYTHONPATH:-}"

# Change to VLAb directory
cd /fsx/dana_aubakirova/vla/VLAb

echo "=========================================="
echo "Testing VLAb Training Command"
echo "=========================================="
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Accelerate: $(which accelerate)"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""

echo "Checking files:"
echo "  ✓ accelerate config: $(test -f accelerate_configs/single_gpu.yaml && echo 'exists' || echo 'MISSING')"
echo "  ✓ train.py: $(test -f src/lerobot/scripts/train.py && echo 'exists' || echo 'MISSING')"
echo ""

echo "Testing imports:"
python -c "from lerobot.configs.train import TrainPipelineConfig; print('  ✓ TrainPipelineConfig')" 2>&1
python -c "from lerobot.policies.factory import make_policy; print('  ✓ Policy factory')" 2>&1
python -c "from lerobot.datasets.factory import make_dataset; print('  ✓ Dataset factory')" 2>&1
echo ""

echo "Testing accelerate config:"
accelerate env 2>&1 | head -5
echo ""

echo "Validating command structure:"
echo "  Command: accelerate launch --config_file accelerate_configs/single_gpu.yaml \\"
echo "    src/lerobot/scripts/train.py \\"
echo "    --policy.type=smolvla2 \\"
echo "    --dataset.repo_id=\"Beegbrain/pick_lemon_and_drop_in_bowl,Chojins/chess_game_000_white_red\" \\"
echo "    --dataset.video_backend=pyav \\"
echo "    --output_dir=\"./outputs/training\" \\"
echo "    --batch_size=8 \\"
echo "    --steps=200000 \\"
echo "    --wandb.enable=true \\"
echo "    --wandb.project=\"smolvla2-training\""
echo ""
echo "=========================================="
echo "To run the actual training, execute:"
echo "  accelerate launch --config_file accelerate_configs/single_gpu.yaml \\"
echo "    src/lerobot/scripts/train.py \\"
echo "    --policy.type=smolvla2 \\"
echo "    --dataset.repo_id=\"Beegbrain/pick_lemon_and_drop_in_bowl,Chojins/chess_game_000_white_red\" \\"
echo "    --dataset.video_backend=pyav \\"
echo "    --output_dir=\"./outputs/training\" \\"
echo "    --batch_size=8 \\"
echo "    --steps=200000 \\"
echo "    --wandb.enable=true \\"
echo "    --wandb.project=\"smolvla2-training\""
echo "=========================================="

