#!/bin/sh

#SBATCH --account=compsci 
#SBATCH --job-name="POL_ID_Full_Pipeline" 
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL 
                                     
# Resource Allocation
#SBATCH --account=l40sfree 
#SBATCH --partition=l40s 
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1 
#SBATCH --time=48:00:00 

# Output and error files
#SBATCH --output=full_pipeline_output_%j.out
#SBATCH --error=full_pipeline_error_%j.err

# Environment Setup
echo "------------------------------------------------------------------------"
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "SLURM Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "SLURM CPUs per Task: $SLURM_CPUS_PER_TASK" 
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------------------------"

# Load necessary modules
echo "Loading modules..."
module purge 
module load python/miniconda3-py3.9 
echo "Modules loaded."

echo "------------------------------------------------------------------------"
echo "Python version:"
python --version
echo "Conda prefix: $CONDA_PREFIX"
echo "------------------------------------------------------------------------"

# Activate custom Conda environment
echo "Activating Conda environment: full_pipeline_env"
source activate full_pipeline_env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment."
    exit 1
fi
echo "Conda environment activated. Current Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "------------------------------------------------------------------------"

# Navigate to Script Directory
TARGET_DIR=~/pol_id/full_pipeline
echo "Navigating to script directory: $TARGET_DIR"
cd $TARGET_DIR
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to $TARGET_DIR."
    exit 1
fi
echo "Current directory after cd: $(pwd)"
echo "------------------------------------------------------------------------"

# Running the Pipeline Dispatcher Script
PYTHON_SCRIPT="run_all_pipelines.py"
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
