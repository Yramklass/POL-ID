#!/bin/bash

#==============================================================================
# SBATCH SCRIPT FOR TRAINING AN RTMDET MODEL
#
# Description:
#   This script submits a job to Slurm to train an MMDetection model (e.g.,
#   RTMDet) for pollen detection. It activates the necessary Conda environment
#   and uses MMDetection's standard tools to start training with a specified
#   configuration file (e.g. configs/rtmdet.py).
#
# Usage:
#   sbatch run_rtmdet_model.sbatch
#
# Notes:
#   - This script assumes you are submitting the job from your project's
#     root directory (e.g., '~/pol_id/detection/').
#   - Ensure the Conda environment 'mmdetection_env' is correctly set up.
#     This environment can be created using 'mmdetection_environment.yaml'.
#==============================================================================

# Configuration
CONDA_ENV_NAME="mmdetection_env"
# Path to your MMDetection training configuration file, relative to the
# project root where you submit the job.
CONFIG_FILE="configs/rtmdet.py"
# The root directory of your project, where MMDetection's tools are located.
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"


# SBATCH Directives 
#SBATCH --job-name="MMDet-Pollen-Train"
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=example@mail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=mmdet_pollen_train_%j.out
#SBATCH --error=mmdet_pollen_train_%j.err


# Job Setup & Environment 
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Load necessary modules
echo "Loading modules..."
module purge
module load python/miniconda3-py3.9

# Activate Conda environment
echo "Activating Conda environment: $CONDA_ENV_NAME"
source activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment '$CONDA_ENV_NAME'."
    exit 1
fi
echo "Python path: $(which python)"


# Script Execution 
# Navigate to the MMDetection project root directory
echo "Navigating to MMDetection project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "ERROR: Failed to navigate to $PROJECT_ROOT."; exit 1; }
echo "Current directory after cd: $(pwd)"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at: $(pwd)/$CONFIG_FILE"
    exit 1
fi

# Run the MMDetection training script
echo "Starting MMDetection training..."
echo "Using config file: $CONFIG_FILE"
echo "Start time: $(date)"

python mmdetection/tools/train.py "$CONFIG_FILE"

# Capture and report the exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "MMDetection training script completed successfully."
else
    echo "ERROR: MMDetection training script exited with status $EXIT_STATUS."
fi

echo "End time: $(date)"
echo "Job finished."