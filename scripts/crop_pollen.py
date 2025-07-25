import os
import json
import random
from PIL import Image
from sklearn.model_selection import train_test_split # For stratified splitting
import shutil # For cleaning up output directory
import numpy as np 

# CONFIGURATION 
script_directory = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(script_directory, "..", "data", "classification"))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_crops")
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15} 
SEED = 42 # For reproducible splits

# Initialize random seed
random.seed(SEED)

def group_stacks(image_files):
    """
    Groups image filenames into stacks. A stack ends when a filename contains "box".
    Assumes image_files are from a single taxon and sorted if specific order within stack matters before this.
    """
    stacks = []
    current_stack = []
    # Sort files to ensure consistent stack grouping if not already sorted
    # This is important if "box" isn't always the absolute last or if order matters
    sorted_image_files = sorted(image_files) 

    for fname in sorted_image_files:
        
        if "box" in fname.lower(): # Assuming "box" marks the end of a distinct slide/stack
            stacks.append(list(current_stack)) # Add a copy of the current stack
            current_stack = [] # Reset for the next stack
        else:
            current_stack.append(fname)
    
    # Add any remaining files as the last stack if it's not empty
    # This handles cases where the last sequence might not end with a "box" file,
    # or if there's only one stack without a "box" file.
    if current_stack:
        stacks.append(list(current_stack))
    return stacks

def slugify(name):
    """
    Simple slugify function to create safe directory/file names.
    """
    name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    return "".join(c for c in name if c.isalnum() or c == '_').strip('_')


def crop_and_save(image_path, annotations_for_image, output_base_dir, image_name, id_to_category_map):
    """
    Crops objects from a single image based on its annotations and saves them.
    Args:
        image_path (str): Path to the source image.
        annotations_for_image (list): List of annotation objects for this specific image.
        output_base_dir (str): The base directory for the current split (e.g., OUTPUT_DIR/train).
        image_name (str): The filename of the source image.
        id_to_category_map (dict): Mapping from category_id to category_name for this annotation set.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Warning: Image file not found {image_path}, skipping.")
        return
    except Exception as e:
        print(f"Warning: Could not open image {image_path}, error: {e}, skipping.")
        return

    for ann_idx, ann in enumerate(annotations_for_image):
        try:
            x, y, w, h = map(int, ann["bbox"])
            if w <= 0 or h <= 0:
                print(f"Warning: Invalid bbox dimensions (w={w}, h={h}) for annotation ID {ann.get('id', 'N/A')} in {image_name}. Skipping crop.")
                continue
            
            cropped = image.crop((x, y, x + w, y + h))
            
            label_id = ann["category_id"]
            # Use the provided id_to_category_map from the specific annotation file
            raw_label_name = id_to_category_map.get(label_id, f"unknown_category_{label_id}")
            label_name_slug = slugify(raw_label_name)

            label_dir = os.path.join(output_base_dir, label_name_slug)
            os.makedirs(label_dir, exist_ok=True)
            
            # Create a unique name for the crop
            crop_name = f"{os.path.splitext(image_name)[0]}_crop{ann.get('id', ann_idx)}.jpg"
            cropped.save(os.path.join(label_dir, crop_name))
        except Exception as e:
            print(f"Error processing annotation {ann.get('id', 'N/A')} for image {image_name}: {e}")


def main():
    # Clean and create output directories
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    for split_name in SPLIT_RATIOS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)
    print(f"Created output directory structure in: {OUTPUT_DIR}")

    from collections import defaultdict
    taxon_to_stacks = defaultdict(list)

    print("Collecting image stacks from all taxa...")
    for taxon_folder_name in os.listdir(BASE_DIR):
        taxon_base_path = os.path.join(BASE_DIR, taxon_folder_name)
        if taxon_folder_name == os.path.basename(OUTPUT_DIR) or not os.path.isdir(taxon_base_path):
            continue

        print(f"  Processing taxon: {taxon_folder_name}")
        image_files_in_taxon = [f for f in os.listdir(taxon_base_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        annotation_file_path = os.path.join(taxon_base_path, "instances_default.json")

        if not image_files_in_taxon:
            print(f"    No image files found in {taxon_folder_name}, skipping.")
            continue
        if not os.path.exists(annotation_file_path):
            print(f"    Annotation file 'instances_default.json' not found in {taxon_folder_name}, skipping.")
            continue

        stacks_from_taxon = group_stacks(image_files_in_taxon)
        if not stacks_from_taxon:
            print(f"    No stacks formed for taxon {taxon_folder_name}, skipping.")
            continue

        print(f"    Found {len(stacks_from_taxon)} stacks for taxon {taxon_folder_name}.")

        slug = slugify(taxon_folder_name)
        for stack in stacks_from_taxon:
            taxon_to_stacks[slug].append({
                'files': stack,
                'taxon_name': slug,
                'annotation_file': annotation_file_path,
                'taxon_base_path': taxon_base_path
            })

    # Filter out taxa with < 3 stacks
    taxon_to_stacks = {k: v for k, v in taxon_to_stacks.items() if len(v) >= 3}
    print(f"\nTotal valid taxa with >= 3 stacks: {len(taxon_to_stacks)}")

    # Split each taxon into train/val/test
    train_stacks, val_stacks, test_stacks = [], [], []
    from math import floor

    for taxon, stacks in taxon_to_stacks.items():
        random.shuffle(stacks)
        n = len(stacks)

        # Ensure at least 1 stack per split
        n_train = max(1, floor(n * SPLIT_RATIOS['train']))
        n_val = max(1, floor(n * SPLIT_RATIOS['val']))
        n_test = n - (n_train + n_val)

        # If test gets < 1, borrow from val (or train)
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1

        # Final safeguard if still imbalanced
        total = n_train + n_val + n_test
        if total < n:
            n_train += (n - total)  # Priority: train > test > val

        # Assign
        train_stacks += stacks[:n_train]
        val_stacks += stacks[n_train:n_train + n_val]
        test_stacks += stacks[n_train + n_val:n_train + n_val + n_test]

        # Debug print per-taxon split
        print(f"Taxon '{taxon}': total {n} → train: {n_train}, val: {n_val}, test: {n_test}")

    splits_map = {
        "train": train_stacks,
        "val": val_stacks,
        "test": test_stacks
    }
    def count_taxa_in_split(stacks_list):
        return set(stack['taxon_name'] for stack in stacks_list)

    print("\n--- Taxon Coverage per Split ---")
    print(f"Train: {len(count_taxa_in_split(train_stacks))} taxa")
    print(f"Val:   {len(count_taxa_in_split(val_stacks))} taxa")
    print(f"Test:  {len(count_taxa_in_split(test_stacks))} taxa")
    print("---------------------------------")


    print("\nProcessing and saving crops for each split...")
    for split_name, stacks_in_split in splits_map.items():
        print(f"  Processing {split_name} set ({len(stacks_in_split)} stacks)...")
        if not stacks_in_split: 
            print(f"    Split '{split_name}' is empty, skipping.")
            continue
            
        output_subdir_for_split = os.path.join(OUTPUT_DIR, split_name)

        for stack_info in stacks_in_split:
            stack_files = stack_info['files']
            annotation_file = stack_info['annotation_file']
            taxon_base_path = stack_info['taxon_base_path'] 

            try:
                with open(annotation_file, 'r') as f:
                    coco_data = json.load(f)
            except Exception as e:
                print(f"    Error loading annotation file {annotation_file}: {e}. Skipping stack.")
                continue
            
            id_to_category_map = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
            image_id_map = {os.path.basename(img["file_name"]): img["id"] for img in coco_data.get("images", [])}
            annotations_by_img_id = {}
            for ann in coco_data.get("annotations", []):
                annotations_by_img_id.setdefault(ann["image_id"], []).append(ann)

            for image_filename_in_stack in stack_files:
                image_id = image_id_map.get(image_filename_in_stack)
                if image_id is None:
                    print(f"    Warning: Image {image_filename_in_stack} not found in annotation JSON. Skipping.")
                    continue

                annotations_for_current_image = annotations_by_img_id.get(image_id, [])
                if not annotations_for_current_image:
                    continue 

                image_full_path = os.path.join(taxon_base_path, image_filename_in_stack)
                crop_and_save(image_full_path, annotations_for_current_image, 
                              output_subdir_for_split, image_filename_in_stack, id_to_category_map)

    print("\nCropping and splitting process complete.")
    print_dataset_summary(OUTPUT_DIR)



def print_dataset_summary(output_dir):
    print("\n--- Dataset Summary ---")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        print(f"\nSplit: {split}")
        if not os.path.isdir(split_path):
            print("  Directory not found.")
            continue
        
        total_images_in_split = 0
        taxon_counts = {}
        for taxon_name_slug in sorted(os.listdir(split_path)):
            taxon_path = os.path.join(split_path, taxon_name_slug)
            if os.path.isdir(taxon_path):
                num_images = len([f for f in os.listdir(taxon_path) if f.lower().endswith('.jpg')])
                taxon_counts[taxon_name_slug] = num_images
                total_images_in_split += num_images
        
        for taxon_name_slug, num_images in sorted(taxon_counts.items()):
             print(f"  Taxon: {taxon_name_slug}, Images: {num_images}")
        print(f"  Total images in {split}: {total_images_in_split}")
    print("-----------------------")


if __name__ == "__main__":
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: scikit-learn is required for stratified splitting. Please install it (pip install scikit-learn).")
        exit()
    
    if not np.isclose(sum(SPLIT_RATIOS.values()), 1.0): 
        print(f"Error: SPLIT_RATIOS must sum to 1.0. Current sum: {sum(SPLIT_RATIOS.values())}")
        exit()

    main()
