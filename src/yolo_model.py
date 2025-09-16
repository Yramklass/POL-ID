"""
yolo_model.py

Description:
    Trains and evaluates a YOLO object detection model for pollen grain 
    detection using the `ultralytics` framework. 

    This script provides a streamlined workflow:
    1.  Initializes a pre-trained YOLOv8 model (e.g., 'yolov8s.pt').
    2.  Trains the model on a custom dataset specified by a .yaml file.
    3.  After training, it automatically identifies the best model checkpoint.
    4.  Runs a final evaluation on the test set to compute and display key 
        metrics like mAP, precision, and recall.

    All configurations, including paths and hyperparameters, are set as global 
    variables at the top of the script.

Usage:
    # For direct execution
    python yolo_model.py

    # For submitting the job to the Slurm workload manager
    sbatch run_YOLO_model.sbatch

Inputs:
    - Dataset YAML file (set in script: DATASET_YAML_PATH): A .yaml file in 
      the YOLO format that defines the paths to train/validation/test image 
      sets and lists the class names.
    - Output directory path (set in script: OUTPUT_PARENT_DIR)
    - Pre-trained model choice (set in script: MODEL_CHOICE): The base YOLOv8 
      model checkpoint to start training from (e.g., 'yolov8s.pt').

Outputs:
    - YOLO run directory (path set by OUTPUT_PATH and RUN_NAME): A comprehensive 
      output folder created by the ultralytics library, containing:
        - Model checkpoints in the 'weights/' subdirectory (including best.pt).
        - Training and validation metrics plots (e.g., results.png, confusion_matrix.png).
        - Logs, event files, and detailed validation results.
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import os

# Configuration
# Path Definitions 

# Path to dataset YAML file
DATASET_YAML_PATH = Path('path/to/yaml_file')
# Path to output directory
OUTPUT_PARENT_DIR = Path('path/to/output/directory') 

RUN_NAME = "yolov8_run1"
OUTPUT_PATH = OUTPUT_PARENT_DIR / RUN_NAME

# Model & Training Hyperparameters
MODEL_CHOICE = 'yolov8s.pt'  # Model type and size can be modified - e.g. yolo11 or yolov8 - 'n'/'s'/'m'/'l' for nano/small/medium/large
EPOCHS = 100                 # Number of training epochs
PATIENCE = 30
IMAGE_SIZE = 640             # Target image size for training
BATCH_SIZE = 16              # Number of images per batch (-1 for auto-batch)


def main():
    """
    Main function to run the YOLO training and evaluation pipeline.
    """
    print("YOLO Pollen Detector Training")
    
    # Setup and Pre-checks
    
    # Check if the dataset YAML file exists
    if not DATASET_YAML_PATH.exists():
        print(f"ERROR: Dataset YAML file not found at: {DATASET_YAML_PATH}")
        print("Please ensure you have run the data preparation script and the file exists.")
        return

    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create the output directory if it doesn't exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"All outputs will be saved to: {OUTPUT_PATH / RUN_NAME}")
    
    
    # MODEL TRAINING 
    # Initialize a YOLO model from a pretrained checkpoint
    print(f" initializing model with '{MODEL_CHOICE}'...")
    model = YOLO(MODEL_CHOICE)
    model.to(device)

    print("\nStarting model training...")
    # The 'train' method returns a results object with training metrics and paths
    results = model.train(
        data=str(DATASET_YAML_PATH),
        epochs=EPOCHS,
        patience=PATIENCE, 
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(OUTPUT_PATH),  # Sets the parent directory for output
        name=RUN_NAME,             # Sets the specific folder name for this run
        exist_ok=True              # Allows overwriting of a previous run with the same name
    )
    print("Training complete.")

    
    # Evaluation on Test Set
    
    print("\nPerforming final evaluation on the test set...")
    
    # Path to the best performing model's weights, saved during training
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    
    if not best_model_path.exists():
        print(f"ERROR: Best model weights not found at '{best_model_path}'.")
        print("Skipping final evaluation.")
        return
        
    # Load the best model for evaluation
    best_model = YOLO(best_model_path)
    
    # Run validation on the 'test' split defined in the YAML file
    metrics = best_model.val(
        split='test',
        data=str(DATASET_YAML_PATH)
    )
    
    print("\n--- Test Set Performance ---")
    print(f"  mAP50-95 (Box): {metrics.box.map:.4f}")
    print(f"  mAP50 (Box):    {metrics.box.map50:.4f}")
    print(f"  Precision (Box): {metrics.box.p[0]:.4f}") # Precision for the 'pollen' class
    print(f"  Recall (Box):    {metrics.box.r[0]:.4f}")    # Recall for the 'pollen' class
    print("----------------------------\n")
    print(f"Evaluation metrics and plots saved in: {metrics.save_dir}")
    print("Full process complete.")


if __name__ == '__main__':
    main()