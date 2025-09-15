#!/bin/sh

#SBATCH --account=compsci
#SBATCH --job-name="TrainDEIMModel"
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# Resource Allocation
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=48:00:00

# Output and error files
#SBATCH --output=detector_train_output_%j.out
#SBATCH --error=detector_train_error_%j.err

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
module load python/miniconda3-py3.11
echo "Modules loaded."

# Create Conda environment 
echo "Creating Conda environment: deim"
conda create -n deim python=3.11.9
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create Conda environment."
    exit 1
fi
echo "Conda environment created."


# Activate Conda environment
echo "Activating Conda environment: deim"
source activate deim
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Conda environment activated. Current Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"


# Install Pip dependencies to user account
pip install --user torch>=2.0.1
pip install --user torchvision>=0.15.2
pip install --user faster-coco-eval>=1.6.5
pip install --user PyYAML
pip install --user tensorboard
pip install --user scipy
pip install --user calflops
pip install --user transformers
pip install --user git+https://github.com/dnth/DEIM.git@v0.2.1


# DEIMKIT - Python Wrapper for DEIM Training
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7778 --nproc_per_node=1 deimkit_training.py

echo "Training finished."