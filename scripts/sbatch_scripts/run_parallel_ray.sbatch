#!/bin/sh

#==============================================================================
# SBATCH SCRIPT FOR RUNNING THE RAYTUNE PARALLEL FUSION TRAINING SCRIPT
#
# Description:
#   This script submits a job to the Slurm workload manager to run Raytune parallel 
#   fusion training and evaluation script. It sets up the required
#   Conda environment, navigates to the project directory, and executes
#   the main dispatcher script 'parallel_fusion_raytune.py'.
#
# Usage:
#   sbatch run_parallel_fusion_ray.sbatch
#
# Notes:
#   Ensure the Conda environment 'classification_env' exists and has all
#   necessary packages installed. This environment can be created using
#   'classification_environment.yaml'.
#==============================================================================

# SBATCH Directives
#SBATCH --account=compsci 
#SBATCH --job-name="POL_ID_RayTune_HPO" 
#SBATCH --mail-user=example@mail.com
#SBATCH --mail-type=BEGIN,END,FAIL 
                                     
# Resource Allocation 
#SBATCH --account=l40sfree 
#SBATCH --partition=l40s 
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1 
#SBATCH --time=48:00:00  

# Output and error files
#SBATCH --output=raytune_hpo_output_%j.out
#SBATCH --error=raytune_hpo_error_%j.err

# Configuration
# Define the name of your Conda environment
CONDA_ENV_NAME="classification_env"
# Define the main Python script to be executed
PYTHON_SCRIPT="parallel_fusion_model.py"
# The script will run in the directory where you submit the job
# (no need to hardcode a path). If your script is elsewhere, define a relative path.
# For example: TARGET_DIR="${SLURM_SUBMIT_DIR}/src"
TARGET_DIR="${SLURM_SUBMIT_DIR}"
# Path to output directory
RESULTS_DIR="./ray_results"

# Environment Setup
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "SLURM Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "SLURM CPUs per Task: $SLURM_CPUS_PER_TASK" 
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"

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


# Set Ray environment variables for HPC
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export TUNE_DISABLE_AUTO_CALLBACK_LOGGERS=1  # Reduce logging overhead

# Set temporary directory for Ray 
export RAY_TMPDIR=$SLURM_TMPDIR
if [ -z "$RAY_TMPDIR" ]; then
    export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
    mkdir -p $RAY_TMPDIR
fi
echo "Ray temporary directory: $RAY_TMPDIR"


# Navigate to Script Directory
echo "Navigating to script directory: $TARGET_DIR"
cd $TARGET_DIR
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"


# Create results directory
mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"


# Running Ray Tune Hyperparameter Optimization
echo "Running Ray Tune hyperparameter optimization: $PYTHON_SCRIPT"
echo "Start time: $(date)"

# Set CUDA memory management for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python $PYTHON_SCRIPT

# Capture exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Ray Tune hyperparameter optimization completed successfully."
    
    # Show best results
    
    echo "BEST HYPERPARAMETERS FOUND:"
    if [ -f "best_hyperparameters.txt" ]; then
        cat best_hyperparameters.txt
    fi
    
    # Compress results for easy transfer
    tar -czf ray_results_${SLURM_JOB_ID}.tar.gz $RESULTS_DIR
    echo "Results compressed to: ray_results_${SLURM_JOB_ID}.tar.gz"
    
else
    echo "ERROR: Ray Tune script exited with status $EXIT_STATUS."
fi

# Cleanup temporary Ray directory
if [ ! -z "$RAY_TMPDIR" ] && [ -d "$RAY_TMPDIR" ]; then
    echo "Cleaning up Ray temporary directory: $RAY_TMPDIR"
    rm -rf $RAY_TMPDIR
fi


echo "End time: $(date)"
echo "Job finished."
