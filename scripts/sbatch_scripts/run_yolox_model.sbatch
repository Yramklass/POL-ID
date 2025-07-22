#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="YOLOX-Pollen-Train"
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --output=mmdet_pollen_train_%j.out
#SBATCH --error=mmdet_pollen_train_%j.err

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=== Job started at $(date) ==="
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# 1. Clean Module and Environment Setup
echo "--- Setting up modules and environment ---"
module purge
module load python/miniconda3-py3.9

# 2. Activate Existing Conda Environment
ENV_NAME="yolox_mmdet_env"
source activate $ENV_NAME
echo "Activated Conda environment: $ENV_NAME"

# 3. Set Python Paths for User Installations
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"

# 4. Verify Critical Packages and GPU Access
echo "--- Verifying packages ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()} | CUDA version: {torch.version.cuda}')"


# 6. Define Absolute Paths for Robust Execution
MMDET_DIR="/home/rmkyas002/pol_id/detection/mmdetection"
CONFIG_FILE="/home/rmkyas002/pol_id/detection/configs/yolox.py"
WORK_DIR="/scratch/rmkyas002/mmdet_outputs/pollen_yolox"

# Add a check to ensure the config file exists before trying to train
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "--- Starting MMDetection training ---"
echo "Using config: $CONFIG_FILE"
echo "Outputting to work directory: $WORK_DIR"

# 7. Run Training
# Prepending PYTHONPATH ensures the local mmdetection repo is found first.
PYTHONPATH="$MMDET_DIR:$PYTHONPATH" \
python -u $MMDET_DIR/tools/train.py \
    $CONFIG_FILE \
    --work-dir $WORK_DIR \
    --cfg-options \
        randomness.seed=42 \
        train_dataloader.persistent_workers=True \
    --launcher="none" \
    --auto-scale-lr

echo "=== Job finished successfully at $(date) ==="