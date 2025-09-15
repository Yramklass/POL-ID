"""
run_all_pipelines.py

Description:
    Runs full honey analysis pipeline for all honey samples.

Usage:
    # For direct execution
    python run_all_pipelines.py

    # For submitting the job to the Slurm workload manager
    sbatch run_all_pipelines.sbatch
    
Inputs:
    - Sample inputs directory path (set in script: sample_inputs_root)
    - Full pipeline script path (set in script: full_pipeline_script)
"""

import os
import subprocess

# Path to inputs directory
sample_inputs_root = 'path/to/directory'

# Path to the full_pipeline script
full_pipeline_script = 'path/to/full_pipeline.py'

# Get all subdirectories in sample_inputs
for entry in os.listdir(sample_inputs_root):
    folder_path = os.path.join(sample_inputs_root, entry)
    if os.path.isdir(folder_path):
        print(f"\nRunning pipeline for: {folder_path}")
        try:
            # Call: python full_pipeline.py <folder_path>
            result = subprocess.run(
                ["python", full_pipeline_script, folder_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline for {folder_path}")
            print(e.stderr)

