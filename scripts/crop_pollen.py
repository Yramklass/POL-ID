import os
import json
import random
from PIL import Image

# CONFIGURATION 
script_directory = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(script_directory, "..", "data", "classification"))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_crops")
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 42

random.seed(SEED)

def group_stacks(image_files):
    stacks = []
    current_stack = []
    for fname in sorted(image_files):
        current_stack.append(fname)
        if "box" in fname.lower():
            stacks.append(current_stack)
            current_stack = []
    if current_stack:
        stacks.append(current_stack)
    return stacks

def slugify(name):
    return name.replace(" ", "_").replace("/", "_")

def crop_and_save(image_path, annotations, output_subdir, image_id, image_name, id_to_category):
    image = Image.open(image_path)
    for ann in annotations:
        if ann["image_id"] == image_id:
            x, y, w, h = map(int, ann["bbox"])
            cropped = image.crop((x, y, x + w, y + h))
            label_id = ann["category_id"]
            label_name = slugify(id_to_category.get(label_id, f"unknown_{label_id}"))
            label_dir = os.path.join(output_subdir, label_name)
            os.makedirs(label_dir, exist_ok=True)
            crop_name = f"{os.path.splitext(image_name)[0]}_crop{ann['id']}.jpg"
            cropped.save(os.path.join(label_dir, crop_name))

def main():
    for taxon_folder in os.listdir(BASE_DIR):
        taxon_path = os.path.join(BASE_DIR, taxon_folder)
        if taxon_folder == "processed_crops" or not os.path.isdir(taxon_path):
            continue

        image_files = [f for f in os.listdir(taxon_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        annotation_path = os.path.join(taxon_path, "instances_default.json")
        if not os.path.exists(annotation_path):
            continue

        with open(annotation_path) as f:
            coco = json.load(f)

        id_to_category = {cat["id"]: cat["name"] for cat in coco["categories"]}
        image_id_map = {img["file_name"]: img["id"] for img in coco["images"]}

        annotations_by_image = {}
        for ann in coco["annotations"]:
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        stacks = group_stacks(image_files)
        random.shuffle(stacks)

        n = len(stacks)
        train_cut = int(n * SPLIT_RATIOS["train"])
        val_cut = int(n * (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]))
        splits = {
            "train": stacks[:train_cut],
            "val": stacks[train_cut:val_cut],
            "test": stacks[val_cut:]
        }

        for split_name, split_stacks in splits.items():
            for stack in split_stacks:
                for image_name in stack:
                    image_id = image_id_map.get(image_name)
                    if image_id is None:
                        continue
                    image_path = os.path.join(taxon_path, image_name)
                    output_subdir = os.path.join(OUTPUT_DIR, split_name)
                    crop_and_save(image_path, coco["annotations"], output_subdir, image_id, image_name, id_to_category)

if __name__ == "__main__":
    main()
