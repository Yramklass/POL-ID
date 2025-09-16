"""
full_pipeline.py

Description:
    End-to-end pipeline for pollen grain analysis from slide images. 
    The script detects pollen grains using a YOLO detector, classifies them 
    with a parallel fusion model (ConvNeXt + Swin Transformer), clusters 
    low-confidence samples with HDBSCAN + UMAP, and generates reports.

Usage:
    python full_pipeline.py <slides_directory>
    # For submitting the job to the Slurm workload manager
    sbatch run_full_pipeline.sbatch

Inputs:
    - slides_directory (command-line argument: path to classification images)
    - YOLO model checkpoint (set in script: YOLO_MODEL_PATH)
    - Classifier model checkpoint (set in script: CLASSIFIER_MODEL_PATH)
    - Base output directory (set in script: base_output_dir)
    - Detector confidence threshold (set in script: DETECTOR_CONF_THRESHOLD)
    - Classifier confidence threshold (set in script: CLASSIFIER_CONFIDENCE_THRESHOLD)
    - Number of pixels to pad detected grains with (set in script: CROP_PADDING)

Outputs:
    - Cropped pollen grain images (per sample)
    - Cluster exemplar images (for low-confidence samples)
    - UMAP visualization plot (if clustering is performed)
    - Composition CSV file for each sample
    - Master clustering metrics CSV (aggregated across runs)
    - Log file capturing stdout for each run
"""

import os
import sys
import csv
import torch
import cv2
import timm
import umap
import hdbscan
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from collections import Counter, defaultdict
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score
from datetime import datetime

# CONFIGURATION
IMG_SIZE = 224
# Path to YOLO model weights
YOLO_MODEL_PATH = 'path/to/yolo_model'
# Path to classifier model weights
CLASSIFIER_MODEL_PATH = 'path/to/classifier_model'
# Path to directory containing honey sample slides
SLIDES_DIR = sys.argv[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds & Parameters
DETECTOR_CONF_THRESHOLD = 0.5 # Higher will mean fewer classified grains 
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.2 # Higher will mean more clustered grains
CROP_PADDING = 10  # Pixels to pad around detected pollen grains

# Get folder name (e.g. "HS095") from input path
input_name = os.path.basename(os.path.normpath(SLIDES_DIR))

# Path to base output directory
base_output_dir = 'path/to/directory'
output_dir = os.path.join(base_output_dir, input_name)
os.makedirs(output_dir, exist_ok=True)

# Redirect stdout to a log file
log_file_path = os.path.join(output_dir, f"{input_name}.out")
sys.stdout = open(log_file_path, 'w')

# MODEL DEFINITION
class ParallelFusionModel(nn.Module):
    def __init__(self, num_classes, convnext_model_name='convnext_small', swin_model_name='swin_small_patch4_window7_224', pretrained=True):
        super().__init__()
        self.convnext_model_name = convnext_model_name
        self.swin_model_name = swin_model_name

        # Load ConvNext backbone
        self.convnext_model = timm.create_model(convnext_model_name, pretrained=pretrained, num_classes=0)
        convnext_feature_dim = self.convnext_model.num_features

        # Load Swin Transformer backbone
        self.swin_model = timm.create_model(swin_model_name, pretrained=pretrained, num_classes=0)
        swin_feature_dim = self.swin_model.num_features

        # Fusion layer and classifier
        self.fusion_dim = convnext_feature_dim + swin_feature_dim
        combined_features_dim = self.fusion_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_features_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

        print(f"ParallelFusionModel initialized:")
        print(f"  ConvNext ({convnext_model_name}) output features: {convnext_feature_dim}")
        print(f"  Swin Transformer ({swin_model_name}) output features: {swin_feature_dim}")
        print(f"  Combined fusion features: {self.fusion_dim}")
        print(f"  Classifier for {num_classes} classes.")

    def forward(self, x, return_embeddings=False):
        features_convnext = self.convnext_model(x)
        features_swin = self.swin_model(x)
        fused_features = torch.cat((features_convnext, features_swin), dim=1)
        output = self.fusion_classifier(fused_features)

        if return_embeddings:
            return output, fused_features
        else:
            return output

# SETUP
# Preprocessing transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Load Models
print("\nLoading YOLO detector...")
detector = YOLO(YOLO_MODEL_PATH)

print("Loading ParallelFusion classifier...")
model = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE, weights_only=False)
model.eval()
model.to(DEVICE)
class_names = model.class_names
print(f"Loaded {len(class_names)} classes: {class_names}")

# Storage for results
predictions_summary = Counter()
cluster_summary = defaultdict(int)
low_conf_embeddings = []
low_conf_images = []

# Initialize variables for metrics logging
num_low_conf = 0
num_clustered_grains = 0
num_noise_grains = 0
cluster_to_low_conf_ratio = 0
num_clusters_found = 0
silhouette_avg = None
db_index = None

# INFERENCE PIPELINE
def detect_and_crop(image_path):
    image = cv2.imread(image_path)
    if image is None: return []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector(image_rgb, verbose=False)[0]

    crops = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(image.shape[1], x2 + CROP_PADDING)
        y2 = min(image.shape[0], y2 + CROP_PADDING)
        crop = image_rgb[y1:y2, x1:x2]
        crops.append((crop, conf))
    return crops

print(f"\nRunning detection and classification on slides in: {SLIDES_DIR}")
for file in tqdm(os.listdir(SLIDES_DIR)):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
        continue

    image_path = os.path.join(SLIDES_DIR, file)
    all_crops = detect_and_crop(image_path)

    batch_tensors = []
    batch_images = []

    # Collect all valid crops from the image
    for crop_img, det_conf in all_crops:
        if det_conf >= DETECTOR_CONF_THRESHOLD:
            image_pil = Image.fromarray(crop_img)
            input_tensor = preprocess(image_pil).to(DEVICE)
            batch_tensors.append(input_tensor)
            batch_images.append(crop_img)

    # If there are valid crops, process them as a single batch
    if batch_tensors:
        input_batch = torch.stack(batch_tensors)
        with torch.no_grad():
            logits, embeddings = model(input_batch, return_embeddings=True)
            probs = torch.softmax(logits, dim=1)
            confidences, pred_classes = torch.max(probs, dim=1)

        # Iterate through the batch results
        for i in range(len(confidences)):
            conf = confidences[i].item()
            pred_class = pred_classes[i].item()
            if conf >= CLASSIFIER_CONFIDENCE_THRESHOLD:
                class_name = class_names[pred_class]
                predictions_summary[class_name] += 1
            else:
                embedding = embeddings[i].cpu().numpy().squeeze()
                low_conf_embeddings.append(embedding)
                low_conf_images.append(batch_images[i])

# CLUSTERING OF UNKNOWNS
if low_conf_embeddings:
    print(f"\nFound {len(low_conf_embeddings)} low-confidence grains to cluster.")
    num_low_conf = len(low_conf_embeddings)
    embeddings_array = np.stack(low_conf_embeddings).astype(np.float64)

    print("Running HDBSCAN...")
    hdbscan_min_cluster_size = 5
    hdbscan_min_samples = 5
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric='cosine',
        algorithm='generic',
        prediction_data=True
    )
    labels = clusterer.fit_predict(embeddings_array)

    num_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise_grains = np.sum(labels == -1)
    num_clustered_grains = num_low_conf - num_noise_grains
    if num_low_conf > 0:
        cluster_to_low_conf_ratio = num_clustered_grains / num_low_conf

    print(f"HDBSCAN found {num_clusters_found} clusters and {num_noise_grains} noise points.")

    if num_clusters_found >= 2:
        core_samples_mask = labels != -1
        core_labels = labels[core_samples_mask]
        core_embeddings = embeddings_array[core_samples_mask]
        silhouette_avg = silhouette_score(core_embeddings, core_labels)
        db_index = davies_bouldin_score(core_embeddings, core_labels)
        print("\n--- Clustering Evaluation Metrics ---")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Davies-Bouldin Index: {db_index:.3f}")
        print("-----------------------------------")
    else:
        print("\nCould not calculate clustering metrics: less than 2 clusters found.")

    for label in labels:
        cluster_name = f"cluster_{label}" if label != -1 else "unknown_noise"
        cluster_summary[cluster_name] += 1

    # UMAP Visualization
    if len(low_conf_embeddings) > 5:
        print("Generating UMAP plot for visualization...")
        n_neighbors = min(15, len(embeddings_array) - 1)
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_array)
        
        plt.figure(figsize=(12, 10))
        unique_labels = sorted(list(set(labels)))
        palette = sns.color_palette("deep", len(unique_labels))
        colors = {label: palette[i] if label != -1 else (0.5, 0.5, 0.5) for i, label in enumerate(unique_labels)}
        
        for i, label in enumerate(labels):
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=[colors[label]], s=15, alpha=0.7)
        
        plt.title('UMAP Projection of Low-Confidence Embeddings')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.grid(True)
        plot_path = os.path.join(output_dir, "umap_cluster_visualization.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Cluster visualization saved to {plot_path}")

    # Save Image Exemplars
    print("Saving image exemplars for each cluster...")
    images_by_cluster = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_name = f"cluster_{label}" if label != -1 else "unknown_noise"
        images_by_cluster[cluster_name].append(low_conf_images[i])
    
    for cluster_name, images in images_by_cluster.items():
        cluster_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        for i, img_array in enumerate(images[:25]):
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cluster_dir, f"exemplar_{i+1}.png"), img_bgr)
    print(f"Image exemplars saved in: {output_dir}")

# REPORTING & LOGGING
print("\n===== HONEY COMPOSITION REPORT =====")
total_grains = sum(predictions_summary.values()) + sum(cluster_summary.values())
if total_grains == 0:
    print("No pollen grains were detected or classified.")
else:
    dominant_classes = []
    # Sort items for consistent report order
    sorted_predictions = sorted(predictions_summary.items(), key=lambda item: item[1], reverse=True)
    sorted_clusters = sorted(cluster_summary.items(), key=lambda item: item[1], reverse=True)
    
    for class_name, count in sorted_predictions:
        percent = 100 * count / total_grains
        print(f"{class_name:<20} | Count: {count:<4} | {percent:.1f}%")
        if percent >= 45:
            dominant_classes.append(class_name)
    
    for cluster_name, count in sorted_clusters:
        percent = 100 * count / total_grains
        print(f"{cluster_name:<20} | Count: {count:<4} | {percent:.1f}%")

    honey_type = f"Monofloral {dominant_classes[0]}" if len(dominant_classes) == 1 else "Multifloral"
    print("-" * 40)
    print(f"Total pollen grains analyzed: {total_grains}")
    print(f"Honey type determination: {honey_type}")
    print("=" * 40)

# Generate Composition CSV
csv_path = os.path.join(output_dir, f"{input_name}_composition.csv")
if total_grains > 0:
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Honey", "Taxon", "% Composition"])
        for class_name, count in predictions_summary.items():
            writer.writerow([input_name, class_name, f"{100 * count / total_grains:.1f}"])
        for cluster_name, count in cluster_summary.items():
            writer.writerow([input_name, cluster_name, f"{100 * count / total_grains:.1f}"])
    print(f"Composition CSV saved to: {csv_path}")

# Save clustering metrics to a master CSV
print("Saving clustering metrics to master log...")
metrics_csv_path = os.path.join(base_output_dir, "master_clustering_metrics.csv")
file_exists = os.path.isfile(metrics_csv_path)

metrics_data = {
    "HoneySample": input_name,
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "DetectorConfThreshold": DETECTOR_CONF_THRESHOLD,
    "ClassifierConfThreshold": CLASSIFIER_CONFIDENCE_THRESHOLD,
    "TotalGrains": total_grains,
    "HighConfidenceGrains": sum(predictions_summary.values()),
    "LowConfidenceGrains": num_low_conf,
    "ClusteredGrains": num_clustered_grains,
    "NoiseGrains": num_noise_grains,
    "ClusterToLowConfRatio": f"{cluster_to_low_conf_ratio:.3f}",
    "NumClustersFound": num_clusters_found,
    "SilhouetteScore": f"{silhouette_avg:.3f}" if silhouette_avg is not None else "N/A",
    "DaviesBouldinIndex": f"{db_index:.3f}" if db_index is not None else "N/A",
    "HDB_min_cluster_size": hdbscan_min_cluster_size if 'hdbscan_min_cluster_size' in locals() else "N/A"
}

try:
    with open(metrics_csv_path, mode='a', newline='') as csvfile:
        fieldnames = metrics_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_data)
    print(f"Metrics successfully saved to {metrics_csv_path}")
except Exception as e:
    print(f"Error saving metrics to CSV: {e}")

# Close the redirected stdout and restore
sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Pipeline complete for {input_name}. Check logs and outputs in {output_dir}")