
import torch
from ultralytics import YOLO
from pathlib import Path
import os

# Configuration

# Path Definitions 

# FOR LOCAL RUN
# try:
#     ROOT_DIR = Path(__file__).resolve().parent.parent
# except NameError:
#     # This fallback is for interactive environments like Jupyter notebooks
#     ROOT_DIR = Path(os.getcwd())

# Path to the dataset configuration YAML file
# DATASET_YAML_PATH = ROOT_DIR / "data/detection/pollen_detector/datasets/pollen/pollen.yaml"
# OUTPUT_PATH = ROOT_DIR / "models" / "detection_outputs"

# FOR HPC RUN
DATASET_YAML_PATH = Path("/scratch/rmkyas002/pollen_detector/datasets/pollen/pollen.yaml")

OUTPUT_PARENT_DIR = Path("/scratch/rmkyas002/detection_outputs") 
RUN_NAME = "pollen_yolov8n_run1"
OUTPUT_PATH = OUTPUT_PARENT_DIR / RUN_NAME

# Model & Training Hyperparameters
MODEL_CHOICE = 'yolo11n.pt'  # 'n' for nano, 's'/'m'/'l'/'x' for small/medium/large/extra-large
EPOCHS = 300                 # Number of training epochs
PATIENCE = 30
IMAGE_SIZE = 640             # Target image size for training
BATCH_SIZE = 16              # Number of images per batch (-1 for auto-batch)


def main():
    """
    Main function to run the YOLOv8 training and evaluation pipeline.
    """
    print("--- YOLOv8 Pollen Detector Training ---")
    
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
    print("Full process complete!")


if __name__ == '__main__':
    main()