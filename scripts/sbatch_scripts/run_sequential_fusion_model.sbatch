#!/bin/bash

#==============================================================================
# SBATCH SCRIPT FOR TRAINING THE SEQUENTIAL FUSION (COATNET) MODEL
#
# Description:
#   This script submits a job to Slurm to train the sequential fusion
#   (CoAtNet) model for pollen classification. It activates the specified
#   Conda environment and runs the two-phase training script.
#
# Usage:
#   sbatch run_sequential_fusion_model.sbatch
#
# Notes:
#   - This script assumes it is submitted from the directory containing the
#     'sequential_fusion_model.py' script.
#   - Ensure the Conda environment 'pollen_classification_env' is correctly set up.
#     This environment can be created using 'classification_environment.yaml'.
#==============================================================================

# Configuration
CONDA_ENV_NAME="pollen_classification_env"
PYTHON_SCRIPT="sequential_fusion_model.py"
# The script will run in the directory where you submit the job.
TARGET_DIR="${SLURM_SUBMIT_DIR}"


# SBATCH Directives
#SBATCH --job-name="POL_ID_Sequential_Fusion"
#SBATCH --account=compsci
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=example@mail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=fusion_output_%j.out
#SBATCH --error=fusion_error_%j.err


# Job Setup & Environment 
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"

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
# Navigate to the target script directory
echo "Navigating to script directory: $TARGET_DIR"
cd "$TARGET_DIR" || { echo "ERROR: Failed to navigate to $TARGET_DIR."; exit 1; }
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

echo "Job finished at $(date)"