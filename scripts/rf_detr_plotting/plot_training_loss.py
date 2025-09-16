"""
plot_training_loss.py

Description:
    Plots RFDETR training loss graph

Usage:
    python plot_training_loss.py

Inputs:
    - RFDETR log file path (set in script: log_file_path)

Outputs:
    - Training loss graph for RFDETR model
"""

import json
import matplotlib.pyplot as plt

# Path to RF-DETR training log file (e.g. log.txt)
log_file_path = "/path/to/file" 

# Lists to store the data for plotting
epochs = []
train_losses = []
val_losses = []
val_maps = []

# Open and read the log file line by line
with open(log_file_path, 'r') as f:
    for line in f:
        try:
            # Each line is a complete JSON object
            data = json.loads(line)
            
            # The epoch number is explicitly in the 'epoch' key
            epochs.append(data['epoch'])
            
            # Get the overall training and validation loss
            train_losses.append(data['train_loss'])
            val_losses.append(data['test_loss']) # 'test_loss' is the validation loss during training
            
            # Get the primary validation mAP score.
            val_maps.append(data['test_coco_eval_bbox'][0])

        except json.JSONDecodeError:
            # Skip any blank or malformed lines
            continue

# Create a figure with two subplots, one on top of the other
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Plot 1: Training and Validation Loss
ax1.plot(epochs, train_losses, 'b-o', label='Training Loss')
ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss')
ax1.set_title('Training & Validation Loss over Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

# Plot 2: Validation mAP
ax2.plot(epochs, val_maps, 'g-o', label='Validation mAP')
ax2.set_title('Validation mAP over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mAP @[IoU=0.50:0.95]')
ax2.grid(True)
ax2.legend()

# Display the plots
plt.tight_layout()
plt.show()