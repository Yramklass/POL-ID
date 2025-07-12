import os
import subprocess

# Path to inputs directory
sample_inputs_root = "/home/yash/POL-ID/data/sample_input"

# Path to the full_pipeline script
full_pipeline_script = "/home/yash/POL-ID/src/full_pipeline.py"

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

