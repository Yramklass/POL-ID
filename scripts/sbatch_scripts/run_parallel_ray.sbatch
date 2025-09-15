#!/bin/sh

#SBATCH --account=compsci 
#SBATCH --job-name="POL_ID_RayTune_HPO" 
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL 
                                     
# Resource Allocation 
#SBATCH --account=l40sfree 
#SBATCH --partition=l40s 
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1 
#SBATCH --time=48:00:00  # Extended time for hyperparameter search

# Output and error files
#SBATCH --output=raytune_hpo_output_%j.out
#SBATCH --error=raytune_hpo_error_%j.err

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
echo "Activating Conda environment: pollen_classification_env"
source activate pollen_classification_env
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
TARGET_DIR=~/pol_id/classification/src/parallel_fusion
echo "Navigating to script directory: $TARGET_DIR"
cd $TARGET_DIR
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"


# Create results directory
RESULTS_DIR="./ray_results"
mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"


# Running Ray Tune Hyperparameter Optimization
PYTHON_SCRIPT="parallel_fusion_raytune.py"  
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
