#!/bin/bash

#==============================================================================
# SBATCH SCRIPT FOR TRAINING RF-DETR MODEL
#
# Description:
#   This script submits a job to Slurm to train an RF-DETR object detection
#   model. It activates the specified Conda environment, installs necessary
#   pip packages to the user's local directory, navigates to the source code,
#   and launches the training script using 'torchrun'.
#
# Usage:
#   sbatch run_rfdetr_model.sbatch
#
# Notes:
#   - This script assumes the base Conda environment 'rfdetr_env' exists.
#     This environment can be created using 'rf_detr_environment.yaml'.
#   - It installs 'rfdetr' and 'roboflow' via pip to the user's local
#     site-packages. This only needs to be done once per environment.
#==============================================================================

# Configuration
CONDA_ENV_NAME="rfdetr_env"
PYTHON_SCRIPT="train_rf_detr_model.py"
# The script will run in the directory where you submit the job.
# Assumes 'train_rf_detr_model.py' is in the same directory as this sbatch script.
TARGET_DIR="${SLURM_SUBMIT_DIR}"


# SBATCH Directives 
#SBATCH --job-name="TrainRFDETRModel"
#SBATCH --account=compsci
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=example@mail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=rfdetr_train_output_%j.out
#SBATCH --error=rfdetr_train_error_%j.err


# Job Setup & Environment 
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Load necessary modules
echo "Loading modules..."
module purge
module load python/miniconda3-py3.g

# Activate Conda environment
echo "Activating Conda environment: $CONDA_ENV_NAME"
source activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Python path: $(which python)"

# Install pip dependencies to user account
echo "Installing required pip packages..."
pip install --user rfdetr
pip install --user roboflow


# Script Execution 
# Navigate to the target script directory
echo "Navigating to script directory: $TARGET_DIR"
cd "$TARGET_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"

# Run the training script using torchrun for distributed training
echo "Running Python script: $PYTHON_SCRIPT with torchrun"
echo "Start time: $(date)"

torchrun --nproc_per_node=1 "$PYTHON_SCRIPT"

# Capture and report the exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "ERROR: Python script exited with status $EXIT_STATUS."
fi

echo "End time: $(date)"
echo "Job finished."