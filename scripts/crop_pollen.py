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
        current_stack.append(fname)
        if "box" in fname.lower(): # Assuming "box" marks the end of a distinct slide/stack
            stacks.append(list(current_stack)) # Add a copy of the current stack
            current_stack = [] # Reset for the next stack
    
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

    all_stacks_info = [] # To store {'files': stack, 'taxon_name': taxon_folder, 'annotation_file': path, 'taxon_base_path': path}
    all_taxon_labels_for_stratification = [] # Parallel list of taxon_names for stratification

    print("Collecting image stacks from all taxa...")
    for taxon_folder_name in os.listdir(BASE_DIR):
        taxon_base_path = os.path.join(BASE_DIR, taxon_folder_name)
        # Skip the output directory itself and any non-directory files
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

        # Group images from this taxon into stacks
        stacks_from_taxon = group_stacks(image_files_in_taxon)
        if not stacks_from_taxon:
            print(f"    No stacks formed for taxon {taxon_folder_name}, skipping.")
            continue
            
        print(f"    Found {len(stacks_from_taxon)} stacks for taxon {taxon_folder_name}.")

        for stack in stacks_from_taxon:
            all_stacks_info.append({
                'files': stack, # List of image filenames in this stack
                'taxon_name': slugify(taxon_folder_name), # The true taxon this stack belongs to (slugified)
                'annotation_file': annotation_file_path, # Path to its annotation file
                'taxon_base_path': taxon_base_path # Path to the original taxon folder
            })
            all_taxon_labels_for_stratification.append(slugify(taxon_folder_name)) # Use slugified for stratification consistency

    if not all_stacks_info:
        print("No image stacks collected from any taxa. Please check your data structure and BASE_DIR.")
        return

    print(f"\nTotal stacks collected from all taxa: {len(all_stacks_info)}")
    unique_taxa_for_strat = sorted(list(set(all_taxon_labels_for_stratification)))
    print(f"Unique taxa for stratification ({len(unique_taxa_for_strat)}): {', '.join(unique_taxa_for_strat)}")


    # Stratified splitting
    # First, split into (train + val) and test
    # Check if all_taxon_labels_for_stratification is not empty before using np.unique
    if not all_taxon_labels_for_stratification:
        print("Error: No taxon labels available for stratification. Cannot proceed with splitting.")
        return
        
    min_samples_per_class = min(np.unique(all_taxon_labels_for_stratification, return_counts=True)[1])
    
    # Simplified check: if any class has only 1 stack, stratification might be tricky for 3 splits.
    # sklearn's train_test_split needs at least 2 samples (stacks in our case) per class for stratification 
    # if we are creating more than one split from it (e.g., train and test, or train and val).
    if min_samples_per_class < 2 and (SPLIT_RATIOS["val"] > 0 or SPLIT_RATIOS["test"] > 0) : 
         print(f"Warning: Some classes have only {min_samples_per_class} stack(s). Stratification might be imperfect or fail if a split size becomes 0 for a class.")
         print("Consider merging small classes or ensuring each class has at least 2 stacks for robust splitting if val/test sets are desired.")

    print(f"\nSplitting {len(all_stacks_info)} stacks into train/val/test sets...")
    try:
        # Split into (train+val) and test
        # Ensure there are enough samples overall for the first split
        if len(all_stacks_info) < 2 : # train_test_split needs at least 2 samples
            print("Error: Not enough total stacks (<2) to perform train/test split.")
            return

        train_val_stacks_info, test_stacks_info, \
        train_val_labels, _ = train_test_split(
            all_stacks_info, 
            all_taxon_labels_for_stratification,
            test_size=SPLIT_RATIOS["test"],
            stratify=all_taxon_labels_for_stratification,
            random_state=SEED
        )

        # Calculate relative validation size for the second split
        if SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"] == 0: 
            relative_val_size = 0
            if SPLIT_RATIOS["val"] > 0: 
                 print("Warning: val_ratio > 0 but train_ratio + val_ratio = 0. Setting val_size to 0 for second split.")
        else:
            relative_val_size = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])

        if relative_val_size > 0 and len(train_val_stacks_info) > 1 : # train_test_split needs at least 2 samples for the set being split
             # Split (train+val) into train and val
            train_stacks_info, val_stacks_info, _, _ = train_test_split(
                train_val_stacks_info,
                train_val_labels, 
                test_size=relative_val_size,
                stratify=train_val_labels,
                random_state=SEED
            )
        elif len(train_val_stacks_info) > 0: 
            train_stacks_info = train_val_stacks_info
            val_stacks_info = [] 
            if relative_val_size > 0:
                print("Warning: Not enough samples in train_val set to create a validation split, all assigned to train.")
            else:
                print("Validation set size is 0 based on ratios.")
        else: 
            train_stacks_info = []
            val_stacks_info = []


    except ValueError as e:
        print(f"Error during stratified splitting: {e}")
        print("This can happen if a class has too few samples (stacks) for the specified split ratios,")
        print("or if the total number of samples is too small for the splits.")
        print("Ensure each class intended for splitting has at least 2 stacks.")
        return

    splits_map = {
        "train": train_stacks_info,
        "val": val_stacks_info,
        "test": test_stacks_info
    }

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
            image_id_map = {img["file_name"]: img["id"] for img in coco_data.get("images", [])}
            
            annotations_by_img_id = {}
            for ann in coco_data.get("annotations", []):
                annotations_by_img_id.setdefault(ann["image_id"], []).append(ann)

            for image_filename_in_stack in stack_files:
                image_id = image_id_map.get(image_filename_in_stack)
                
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
