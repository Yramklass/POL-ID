"""
train_rf_detr_model.py

Description:
    Initializes and trains a small RF-DETR (Real-time DEtection TRansformer) 
    model for pollen grain object detection. 

    This script acts as a high-level launcher for the training process, which 
    is encapsulated within the `RFDETRSmall` class. It sets the dataset and 
    output directories, instantiates the model, and calls the `.train()` method 
    with a predefined set of hyperparameters (e.g., learning rate, batch size, 
    number of epochs, and early stopping).

Usage:
    # For direct execution (e.g., on a local machine or for testing)
    python train_rf_detr_model.py

    # For submitting the job to the Slurm workload manager
    sbatch run_rfdetr_model.sbatch

Inputs:
    - Dataset directory (set in script: dataset_directory): Path to the 
      training data, which must be in COCO object detection format.
    - Training Hyperparameters (set in script): All training parameters like 
      epochs, batch size, and learning rate are hardcoded in the call to 
      the `model.train()` method.

Outputs:
    - Output directory (set in script: output_directory): This folder will 
      contain all artifacts from the training run, including model checkpoints,
      training logs, and validation metrics.
"""

import os
from rfdetr import RFDETRSmall

def train_pollen_detector():
    # Path to COCO detection dataset
    dataset_directory = 'path/to/coco/dataset' 
    # Path to output directory
    output_directory = 'path/to/output/directory'

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    print(f"Dataset directory: {dataset_directory}")
    print(f"Output will be saved to: {output_directory}")

    # Instantiate the RF-DETR model.
    model = RFDETRSmall()

    # Start the training process.
    model.train(
        dataset_dir=dataset_directory,
        epochs=200,          
        batch_size=8,        
        grad_accum_steps=2,  # Effective batch size = batch_size * grad_accum_steps = 16
        lr=1e-4,            
        output_dir=output_directory,
        world_size=1,
        early_stopping=True,
        early_stopping_patience=50, 
    )
    print("Training complete!")

if __name__ == "__main__":
    train_pollen_detector()