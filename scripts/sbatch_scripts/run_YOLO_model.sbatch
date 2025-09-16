#!/bin/bash

#==============================================================================
# SBATCH SCRIPT FOR TRAINING A ULTRALYTICS YOLO MODEL
#
# Description:
#   This script submits a job to Slurm to train a YOLO object detection
#   model using the ultralytics library. It sets up the Conda environment,
#   installs required pip packages, and executes the training script.
#
# Usage:
#   sbatch run_YOLO_model.sbatch
#
# Notes:
#   - Assumes the Python script is in the same directory where this job is
#     submitted.
#   - Installs 'ultralytics' and 'opencv-python' to the user's local
#     pip directory if not already present.
#   - Ensure the Conda environment 'yolo_detection_env' is correctly set up.
#     This environment can be created using 'yolo_detection_environment.yaml'.
#==============================================================================

# Configuration
CONDA_ENV_NAME="yolo_detection_env"
PYTHON_SCRIPT="yolo_model.py"
# The script will run in the directory where you submit the job.
TARGET_DIR="${SLURM_SUBMIT_DIR}"


# SBATCH Directives
#SBATCH --job-name="TrainYOLOModel"
#SBATCH --account=compsci
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=example@mail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=detector_train_output_%j.out
#SBATCH --error=detector_train_error_%j.err


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
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Python path: $(which python)"

# Install Pip dependencies to user account
echo "Installing required pip packages..."
pip install --user ultralytics
pip install --user opencv-python


# Script Execution 
# Navigate to the target script directory
echo "Navigating to script directory: $TARGET_DIR"
cd "$TARGET_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"

# Run the training script
echo "Running Python script: $PYTHON_SCRIPT"
echo "Start time: $(date)"

python "$PYTHON_SCRIPT"

# Capture and report the exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "ERROR: Python script exited with status $EXIT_STATUS."
fi

echo "End time: $(date)"
echo "Job finished."