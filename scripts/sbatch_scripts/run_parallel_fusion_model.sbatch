#!/bin/sh

#==============================================================================
# SBATCH SCRIPT FOR RUNNING THE PARALLEL FUSION TRAINING SCRIPT
#
# Description:
#   This script submits a job to the Slurm workload manager to run parallel 
#   fusion training and evaluation script. It sets up the required
#   Conda environment, navigates to the project directory, and executes
#   the main dispatcher script 'parallel_fusion_model.py'.
#
# Usage:
#   sbatch run_parallel_fusion_model.sbatch
#
# Notes:
#   Ensure the Conda environment 'classification_env' exists and has all
#   necessary packages installed. This environment can be created using
#   'classification_environment.yaml'.
#==============================================================================

# SBATCH Directives
#SBATCH --account=compsci 
#SBATCH --job-name="POL_ID_Parallel_Fusion" 
#SBATCH --mail-user=example@mail.com
#SBATCH --mail-type=BEGIN,END,FAIL 
                                     
# Resource Allocation
#SBATCH --account=l40sfree 
#SBATCH --partition=l40s 
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1 # Requesting 1 node, 1 task, 1 L40S GPU
#SBATCH --time=48:00:00 # 

# Output and error files
#SBATCH --output=parallel_fusion_output_%j.out
#SBATCH --error=parallel_fusion_error_%j.err

# Configuration
# Define the name of your Conda environment
CONDA_ENV_NAME="classification_env"
# Define the main Python script to be executed
PYTHON_SCRIPT="parallel_fusion_model.py"
# The script will run in the directory where you submit the job
# (no need to hardcode a path). If your script is elsewhere, define a relative path.
# For example: TARGET_DIR="${SLURM_SUBMIT_DIR}/src"
TARGET_DIR="${SLURM_SUBMIT_DIR}"

# Environment Setup
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "SLURM Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "SLURM CPUs per Task: $SLURM_CPUS_PER_TASK" 
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"


# Load necessary modules
echo "Loading modules..."
module purge 
module load python/miniconda3-py3.9 
echo "Modules loaded."

# Activate Conda environment
echo "Activating Conda environment: $CONDA_ENV_NAME"
source activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Conda environment activated. Current Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Navigate to Script Directory
echo "Navigating to script directory: $TARGET_DIR"
cd $TARGET_DIR
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"

# Running Python Script for Parallel Fusion Model
echo "Running Python script: $PYTHON_SCRIPT"
echo "Start time: $(date)"

python $PYTHON_SCRIPT

# Capture exit status of the Python script
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "ERROR: Python script exited with status $EXIT_STATUS."
fi

echo "End time: $(date)"
echo "Job finished."
