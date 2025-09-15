#!/bin/sh

#SBATCH --account=compsci
#SBATCH --job-name="TrainRFDETRModel" 
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# Resource Allocation
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=48:00:00

# Output and error files
#SBATCH --output=rfdetr_train_output_%j.out # <-- Updated Output File
#SBATCH --error=rfdetr_train_error_%j.err   # <-- Updated Error File

# Environment Setup
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"


# Load necessary modules
echo "Loading modules..."
module purge
module load python/miniconda3-py3.9
echo "Modules loaded."


# Activate Conda environment
echo "Activating Conda environment: rfdetr_env"
source activate rfdetr_env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Conda environment activated. Current Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"


# Install pip dependencies to user account
pip install --user rfdetr
# pip install --user roboflow

# Navigate to Script Directory
TARGET_DIR=~/pol_id/detection/src
echo "Navigating to script directory: $TARGET_DIR"
cd $TARGET_DIR
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"


# Running Python Script for RF-DETR Model
PYTHON_SCRIPT="train_rf_detr_model.py"  
echo "Running Python script: $PYTHON_SCRIPT with torchrun"
echo "Start time: $(date)"


torchrun --nproc_per_node=1 $PYTHON_SCRIPT

# Capture exit status of the Python script
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "ERROR: Python script exited with status $EXIT_STATUS."
fi

echo "End time: $(date)"
echo "Job finished."
