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
module load python/miniconda3-py3.11
echo "Modules loaded."
echo "------------------------------------------------------------------------"

# # Remove old Conda environment 
# conda env remove --n yolo_detection_env --yes
# if [ $? -ne 0 ]; then
#     echo "Warning: Failed to remove old Conda environment."
# fi

# # Create Conda environment 
# echo "Creating Conda environment: deim"
# conda create -n deim python=3.11.9
# if [ $? -ne 0 ]; then
#     echo "ERROR: Failed to create Conda environment."
#     exit 1
# fi
# echo "Conda environment created."
# echo "------------------------------------------------------------------------"

# Activate custom Conda environment
echo "Activating Conda environment: deim"
source activate deim
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Conda environment activated. Current Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "------------------------------------------------------------------------"

# # Install pip dependencies to user account
# pip install --user torch>=2.0.1
# pip install --user torchvision>=0.15.2
# pip install --user faster-coco-eval>=1.6.5
# pip install --user PyYAML
# pip install --user tensorboard
# pip install --user scipy
# pip install --user calflops
# pip install --user transformers
pip install --user git+https://github.com/dnth/DEIM.git@v0.2.1

### DEIM LIBRARY TRAINING
# # Training Command 
# # Define variables
# export MASTER_PORT=7777 # Port
# NGPUS=1 # Number of GPUs

# # Path to custom config file
# CONFIG_FILE="$HOME/pol_id/detection/deim/deim_training.yml"

# # Path to pre-trained checkpoint 
# CHECKPOINT_PATH="$HOME/pol_id/detection/deim/deim_dfine_hgnetv2_s_coco_120e.pth"

# echo "Starting training..."
# torchrun --master_port=$MASTER_PORT --nproc_per_node=$NGPUS $HOME/pol_id/detection/deim/DEIM-main/train.py \
#     -c $CONFIG_FILE \
#     --use-amp \
#     --seed=0 \
#     -t $CHECKPOINT_PATH


### DEIMKIT - Python Wrapper for DEIM

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7778 --nproc_per_node=1 deimkit_training.py



echo "Training finished."