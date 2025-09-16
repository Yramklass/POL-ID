#!/bin/bash

#==============================================================================
# SBATCH SCRIPT FOR TRAINING A YOLOX MODEL WITH MMDETECTION
#
# Description:
#   This script submits a job to Slurm to train a YOLOX object detection
#   model using the MMDetection framework. It activates the specific Conda
#   environment, sets necessary Python paths, and launches the training
#   process with a given configuration file, saving outputs to a scratch
#   directory.
#
# Usage:
#   sbatch run_yolox_model.sbatch
#
# Notes:
#   - Assumes you are submitting from your project's root directory.
#   - Ensure the Conda environment 'yolox_mmdet_env' is correctly set up.
#     This environment can be created using 'yolox_environment.yaml'.
#==============================================================================

# Configuration
CONDA_ENV_NAME="yolox_mmdet_env"
# The project root is assumed to be the directory where the job is submitted
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
# Path to the MMDetection library within your project
MMDET_DIR="${PROJECT_ROOT}/mmdetection"
# Path to the model's configuration file
CONFIG_FILE="${PROJECT_ROOT}/configs/yolox.py"
# Path to output directory
WORK_DIR="/path/to/directory"


# SBATCH Directives 
#SBATCH --job-name="TrainYOLOXModel"
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=example@mail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=yolox_pollen_train_%j.out
#SBATCH --error=yolox_pollen_train_%j.err


# Job Setup & Environment 
# Exit immediately if a command exits with a non-zero status.
set -e

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Project Root: $PROJECT_ROOT"

# Clean Module and Environment Setup
module purge
module load python/miniconda3-py3.9

# Activate Conda Environment
source activate "$CONDA_ENV_NAME"
echo "Activated Conda environment: $CONDA_ENV_NAME"

# Set Python Paths for User Installations
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"


# Script Execution 
# Ensure config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "Starting MMDetection training for YOLOX"
echo "Using config: $CONFIG_FILE"
echo "Outputting to work directory: $WORK_DIR"

# Run YOLOX Training
# Prepending PYTHONPATH ensures the local mmdetection repo is found first.
PYTHONPATH="$MMDET_DIR:$PYTHONPATH" \
python -u "$MMDET_DIR/tools/train.py" \
    "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --cfg-options \
        randomness.seed=42 \
        train_dataloader.persistent_workers=True \
    --launcher="none" \
    --auto-scale-lr