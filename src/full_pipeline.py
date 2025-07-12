import os
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import hdbscan
from collections import Counter, defaultdict
from tqdm import tqdm
import torch.nn as nn
import timm
import umap
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys




# Configuration  -- change to allow processing several folders, saving outputs into separate folders, running on HPC
IMG_SIZE = 224  
YOLO_MODEL_PATH = "/home/yash/POL-ID/models/YOLO/yolo_11/nano/epoch_100/detection_outputs/pollen_yolov8n_run1/pollen_yolov8n_run1/weights/best.pt"
CLASSIFIER_MODEL_PATH = "/home/yash/POL-ID/models/par_outputs/75/parallel_fusion/training_outputs_parallel_fusion/pollen_parallel_fusion_final_full.pth"
SLIDES_DIR = sys.argv[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get folder name (e.g. "HS095") from input path
input_name = os.path.basename(os.path.normpath(SLIDES_DIR))

# Define output path
base_output_dir = "/home/yash/POL-ID/outputs/clustering_outputs"
output_dir = os.path.join(base_output_dir, input_name)
os.makedirs(output_dir, exist_ok=True)

# Redirect stdout to output file
log_file_path = os.path.join(output_dir, f"{input_name}.out")
sys.stdout = open(log_file_path, 'w')



# ParallelFusionModel Definition 
class ParallelFusionModel(nn.Module):
    def __init__(self, num_classes, convnext_model_name='convnext_tiny', swin_model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        self.convnext_model_name = convnext_model_name
        self.swin_model_name = swin_model_name

        # Load ConvNext backbone
        # num_classes=0 ensures the classifier head is removed, returning features
        self.convnext_model = timm.create_model(convnext_model_name, pretrained=pretrained, num_classes=0)
        convnext_feature_dim = self.convnext_model.num_features

        # Load Swin Transformer backbone
        self.swin_model = timm.create_model(swin_model_name, pretrained=pretrained, num_classes=0)
        swin_feature_dim = self.swin_model.num_features

        # Fusion layer and classifier
        self.fusion_dim = convnext_feature_dim + swin_feature_dim
        self.fusion_classifier = nn.Linear(self.fusion_dim, num_classes)
        
        combined_features_dim = self.fusion_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_features_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), # <--- ADD STRONG DROPOUT HERE
            nn.Linear(1024, num_classes)
            )

        print(f"ParallelFusionModel initialized:")
        print(f"  ConvNext ({convnext_model_name}) output features: {convnext_feature_dim}")
        print(f"  Swin Transformer ({swin_model_name}) output features: {swin_feature_dim}")
        print(f"  Combined fusion features: {self.fusion_dim}")
        print(f"  Classifier for {num_classes} classes.")
        
    def forward(self, x, return_embeddings=False):
        """
        Modified forward pass.
        - If return_embeddings is False (default), it returns only the final classification logits.
        - If return_embeddings is True, it returns both the logits and the fused feature vector.
        """
        features_convnext = self.convnext_model(x)
        features_swin = self.swin_model(x)
        
        # Feature embedding
        fused_features = torch.cat((features_convnext, features_swin), dim=1)
        
        # Final classification
        output = self.fusion_classifier(fused_features)

        if return_embeddings:
            return output, fused_features
        else:
            return output

CONFIDENCE_THRESHOLD = 0.2  # Below this, trigger clustering
CROP_PADDING = 10  # Pixels to pad around detected pollen grains

# Preprocessing 
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
model = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE)
model.eval()
model.to(DEVICE)
class_names = model.class_names
print(f"Loaded {len(class_names)} classes: {class_names}")

# Storage for Results
predictions_summary = Counter()
low_conf_embeddings = []
low_conf_images = []

# Inference Pipeline 
def detect_and_crop(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector(image_rgb)[0]  # Get the first batch result

    crops = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        # Pad and clip
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
    crops = detect_and_crop(image_path)

    for crop_img, det_conf in crops:
        image_pil = Image.fromarray(crop_img)
        input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, embeddings = model(input_tensor, return_embeddings=True)
            probs = torch.softmax(logits, dim=1)
            conf, pred_class = torch.max(probs, dim=1)
            conf = conf.item()

        if conf >= CONFIDENCE_THRESHOLD:
            class_name = class_names[pred_class.item()]
            predictions_summary[class_name] += 1
        else:
            low_conf_embeddings.append(embeddings.cpu().numpy().squeeze())
            low_conf_images.append(crop_img)

# Cluster Unknown Grains
cluster_summary = defaultdict(int)


if low_conf_embeddings:
    print(f"\nFound {len(low_conf_embeddings)} low-confidence grains to cluster.")
    
    # Prepare Data
    embeddings_array = np.stack(low_conf_embeddings).astype(np.float64)  

    # Run HDBSCAN 
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=5,
        metric='cosine',
        algorithm='generic',
        prediction_data=True
    )
    labels = clusterer.fit_predict(embeddings_array)

    
    # Summarize Results 
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)
    print(f"HDBSCAN found {num_clusters} clusters and {num_noise} noise points.")

    for label in labels:
        cluster_name = f"cluster_{label}" if label != -1 else "unknown_noise"
        cluster_summary[cluster_name] += 1

    # Visualize the Clusters with UMAP 
    print("Generating UMAP plot for visualization...")
    
    # Reduce dimensionality to 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_array)

    # Create a scatter plot
    plt.figure(figsize=(12, 10))
    # Create a palette with a color for each cluster, plus gray for noise
    unique_labels = set(labels)
    palette = sns.color_palette("deep", len(unique_labels))
    # Map labels to colors, making noise (-1) gray
    colors = {label: palette[i] if label != -1 else (0.5, 0.5, 0.5) for i, label in enumerate(unique_labels)}
    
    # Plot each point with its cluster color
    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=[colors[label]], s=10)

    plt.title('UMAP Projection of Pollen Embeddings, Colored by HDBSCAN Cluster')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True)
    plot_path = os.path.join(output_dir, "umap_cluster_visualization.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Cluster visualization saved to {plot_path}")

    # Save Image Exemplars for Each Cluster 
    print("Saving image exemplars for each cluster...")
    images_by_cluster = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_name = f"cluster_{label}" if label != -1 else "unknown_noise"
        images_by_cluster[cluster_name].append(low_conf_images[i])

    for cluster_name, images in images_by_cluster.items():
        cluster_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        for i, img_array in enumerate(images[:25]): # Save up to 25 examples
            # Convert RGB numpy array to BGR for cv2.imwrite
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cluster_dir, f"exemplar_{i+1}.png"), img_bgr)
    print(f"Image exemplars saved in: {output_dir}")


# Final Composition Report 
print("\n===== HONEY COMPOSITION REPORT =====")
total = sum(predictions_summary.values()) + sum(cluster_summary.values())

for class_name, count in predictions_summary.items():
    print(f"{class_name:<20} | Count: {count:<3} | {100 * count / total:.1f}%")

for cluster_name, count in cluster_summary.items():
    print(f"{cluster_name:<20} | Count: {count:<3} | {100 * count / total:.1f}%")

print(f"Total pollen grains analyzed: {total}")
# Optional: Close the redirected stdout to flush file
sys.stdout.close()
sys.stdout = sys.__stdout__
print("Pipeline complete.")
