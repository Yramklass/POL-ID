#!/bin/sh

#SBATCH --account=compsci
#SBATCH --job-name="PollenDetectorTrain"
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# Resource Allocation (modeled after your working script)
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=48:00:00

# Output and error files
#SBATCH --output=detector_train_output_%j.out
#SBATCH --error=detector_train_error_%j.err

# Environment Setup
echo "------------------------------------------------------------------------"
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------------------------"

# Load necessary modules
echo "Loading modules..."
module purge
module load python/miniconda3-py3.9
echo "Modules loaded."
echo "------------------------------------------------------------------------"

# Activate custom Conda environment
echo "Activating Conda environment: pollen_detection_env"
source activate pollen_detection_env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Conda environment activated. Current Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "------------------------------------------------------------------------"

# Navigate to Script Directory
TARGET_DIR=~/pol_id/detection/src
echo "Navigating to script directory: $TARGET_DIR"
cd $TARGET_DIR
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"
echo "------------------------------------------------------------------------"

# Running Python Script for Detector Model
PYTHON_SCRIPT="detector_model.py"
echo "Running Python script: $PYTHON_SCRIPT"
echo "Start time: $(date)"
echo "------------------------------------------------------------------------"

python $PYTHON_SCRIPT

# Capture exit status of the Python script
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "ERROR: Python script exited with status $EXIT_STATUS."
fi
echo "------------------------------------------------------------------------"
echo "End time: $(date)"
echo "Job finished."
echo "------------------------------------------------------------------------"