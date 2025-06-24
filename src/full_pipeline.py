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

# === Configuration ===
IMG_SIZE = 224  # Assumed input size for classifier
YOLO_MODEL_PATH = "../models/YOLO/yolo_11/nano/weights/best.pt"
CLASSIFIER_MODEL_PATH = "../models/par_outputs/75/parallel_fusion/training_outputs_parallel_fusion/pollen_parallel_fusion_final_full.pth"
SLIDES_DIR = "path/to/slides"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = 0.7  # Below this, trigger clustering
CROP_PADDING = 10  # Pixels to pad around detected pollen grains

# === Preprocessing (from training) ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# === Load Models ===
print("\nLoading YOLOv8 detector...")
detector = YOLO(YOLO_MODEL_PATH)

print("Loading ParallelFusion classifier...")
model = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE)
model.eval()
model.to(DEVICE)
class_names = model.class_names
print(f"Loaded {len(class_names)} classes: {class_names}")

# === Storage for Results ===
predictions_summary = Counter()
low_conf_embeddings = []
low_conf_images = []

# === Inference Pipeline ===
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

# === Cluster Unknown Grains ===
cluster_summary = defaultdict(int)
if low_conf_embeddings:
    print("\nRunning HDBSCAN on low-confidence grains...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(np.stack(low_conf_embeddings))

    for label in labels:
        cluster_summary[f"cluster_{label}" if label != -1 else "unknown"] += 1

# === Final Composition Report ===
print("\n===== HONEY COMPOSITION REPORT =====")
total = sum(predictions_summary.values()) + sum(cluster_summary.values())

for class_name, count in predictions_summary.items():
    print(f"{class_name:<20} | Count: {count:<3} | {100 * count / total:.1f}%")

for cluster_name, count in cluster_summary.items():
    print(f"{cluster_name:<20} | Count: {count:<3} | {100 * count / total:.1f}%")

print(f"Total pollen grains analyzed: {total}")
