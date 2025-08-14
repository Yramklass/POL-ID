import json
import matplotlib.pyplot as plt


results_file = '/home/yash/POL-ID/models/rf-detr/run_1/results.json' 

# Load the JSON data
with open(results_file, 'r') as f:
    data = json.load()

epochs = []
map_scores = []

# The keys in results.json are the epoch numbers
for epoch, metrics in data.items():
    # Skip non-epoch entries if any
    if not epoch.isdigit():
        continue
    
    epochs.append(int(epoch))
    # The primary mAP score is under 'bbox' -> 'AP'
    map_scores.append(metrics['bbox']['AP'])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, map_scores, marker='o', linestyle='-', label='Validation mAP@.50:.95')
plt.title('Validation Performance over Epochs')
plt.xlabel('Epoch')
plt.ylabel('mAP @[IoU=0.50:0.95]')
plt.grid(True)
plt.legend()
plt.xticks(epochs) # Ensure all evaluated epochs are marked
plt.tight_layout()
plt.show()