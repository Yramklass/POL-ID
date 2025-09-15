#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --job-name="MMDet-Pollen-Train" 
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --output=mmdet_pollen_train_%j.out 
#SBATCH --error=mmdet_pollen_train_%j.err

# Environment Setup
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"


# Load necessary modules from the HPC
echo "Loading modules..."
module purge
module load python/miniconda3-py3.9
echo "Modules loaded."


# Activate Conda environment for MMDetection
echo "Activating Conda environment: mmdetection_env"
source activate mmdetection_env 
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment 'mmdetection_env'."
    exit 1
fi
echo "Conda environment activated. Current Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"



# MMDetection training is typically run from the root of the mmdetection repo.
MMDET_ROOT=~/pol_id/detection/ 
echo "Navigating to MMDetection root directory: $MMDET_ROOT"
cd $MMDET_ROOT
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $MMDET_ROOT."
    echo "Please ensure you have cloned MMDetection and the path is correct."
    exit 1
fi
echo "Current directory after cd: $(pwd)"



CONFIG_FILE=~/pol_id/detection/configs/rtmdet.py

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at: $CONFIG_FILE"
    exit 1
fi

# Running the MMDetection training script
echo "Starting MMDetection training..."
echo "Using config file: $CONFIG_FILE"
echo "Start time: $(date)"


# The command to start training using MMDetection's tools
python mmdetection/tools/train.py $CONFIG_FILE 

# Capture the exit status of the training script
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "MMDetection training script completed successfully."
else
    echo "ERROR: MMDetection training script exited with status $EXIT_STATUS."
fi

echo "End time: $(date)"
echo "Job finished."
