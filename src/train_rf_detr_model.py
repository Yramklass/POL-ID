import os
from rfdetr import RFDETRSmall

def train_pollen_detector():
    """
    This script trains an RF-DETR model on a custom pollen dataset.
    """

    dataset_directory = "/scratch/rmkyas002/coco_detection_data/" 
    output_directory = "~/pol_id/detection/rf_detr_outputs/" 

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    print(f"Dataset directory: {dataset_directory}")
    print(f"Output will be saved to: {output_directory}")

    # Instantiate the RT-DETR model.
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