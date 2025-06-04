#!/bin/sh

#SBATCH --account=compsci
#SBATCH --job-name="POL_ID_Sequential_Fusion"
#SBATCH --mail-user=rmkyas002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
                         
# Resource Allocation
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=48:00:00

# Output and error files 
#SBATCH --output=fusion_output_%j.out
#SBATCH --error=fusion_error_%j.err

# Environment Setup
echo "Job started on $(hostname) at $(date)"
echo "Current working directory is $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"


# Load necessary modules
module purge 
module load python/miniconda3-py3.9 

echo "Python module loaded. Python version:"
python --version

# Activate custom Conda environment
source activate pollen_classification_env

# Navigate to Script Directory
cd ~/pol_id/scripts/

# Running Python Script
echo "Running Python script: sequential_fusion_model.py"
python sequential_fusion_model.py

# Example if script takes a data path argument:
# python ./scripts/sequential_fusion_model.py --data_path /scratch/rmkyas002/data

echo "Job finished at $(date)"