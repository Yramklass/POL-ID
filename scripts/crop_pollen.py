"""
crop_pollen.py

Description:
    Prepares the final pollen classification dataset from raw, annotated images.
    This script processes a directory of taxon-specific folders, crops individual
    pollen grains based on COCO-style JSON annotations, performs data cleaning,
    and splits the resulting crops into training, validation, and test sets.

    Its primary function is to ensure a clean, well-structured dataset ready
    for input into a deep learning model, while preventing data leakage from
    image stacks.

Key Functionalities:
    - Stack-Level Splitting: To prevent data leakage where different focal
      planes of the same pollen grain end up in both training and test sets,
      the script groups images into "stacks." A stack is defined as a sequence
      of images of the same grain. It identifies the end of a stack by looking
      for a filename containing "_box". The train/validation/test split is
      performed on these stacks, not on individual images or crops.
    - Taxon Normalization and Merging: Cleans up class names by removing "Supp_"
      prefixes and allows for the merging of similar or identical taxa using
      the `TAXON_MERGE_MAP` dictionary.
    - Data Summarization: Generates comprehensive CSV reports and a bar chart
      visualizing the number of stacks and crops per taxon, providing insight
      into the dataset's composition.

Usage:
    python crop_pollen.py

Inputs:
    - A base directory (`BASE_DIR`) containing subdirectories for each pollen
      taxon. Each taxon subdirectory must contain:
        1. The source images (.jpg, .png, etc.).
        2. A corresponding `instances_default.json` file with annotations
           in COCO format.
    - Path where the final structured dataset will be saved (set in script: OUTPUT_DIR)
    - The train, validation, and test split ratios (set in script: SPLIT_RATIOS)
    - A dictionary for consolidating class labels (set in script: TAXON_MERGE_MAP)

Outputs:
    - A new directory (`OUTPUT_DIR`) structured for model training. It contains
      `train/`, `val/`, and `test/` subdirectories, which in turn contain
      folders for each final taxon holding the cropped pollen grain images.
    - Summary reports saved in the `OUTPUT_DIR`:
        - `comprehensive_taxon_summary.csv`: Details counts of original images,
          stacks, and final crops for each taxon.
        - `taxon_stack_counts.csv`: A simple list of stack counts per taxon.
        - `taxon_stack_counts.png`: A bar plot visualizing the stack counts.
"""

import os
import json
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil 
import numpy as np
import csv
import matplotlib.pyplot as plt
import re 
from collections import defaultdict

# Configuration

# Path to classification images directory (in per-taxon folders containing images and label files)
BASE_DIR = 'path/to/directory'

# Path to directory where processed grain crops will be saved
OUTPUT_DIR = 'path/to/directory'

# Dataset split ratios 
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

SEED = 42 # For reproducible splits

# Map for merging taxa.
# Key: original taxon name, Value: new combined taxon name.
TAXON_MERGE_MAP = {
    "PAL0007": "Monocot_sp_5",
    "PAL0008": "Monocot_sp_5"
}

# Initialize random seed
random.seed(SEED)

def normalize_taxon_name(name):
    """
    Normalizes taxon names by removing "Supp" prefixes and extra whitespace.
    e.g., "Supp_Daisy sp. 6" -> "Daisy sp. 6"
    """
    # Case-insensitive removal of "Supp" or "Supp_" at the beginning of the string
    name = re.sub(r'^Supp_?\s*', '', name, flags=re.IGNORECASE)
    return name.strip()

def group_stacks(image_files):
    """
    Groups image filenames into stacks. A stack ends when a filename contains "box".
    Assumes image_files are from a single taxon.
    """
    stacks = []
    current_stack = []
    # Sort files to ensure consistent stack grouping 
    sorted_image_files = sorted(image_files)

    for fname in sorted_image_files:
        if "box" in fname.lower(): # Assuming "box" marks the end of a distinct slide/stack
            stacks.append(list(current_stack)) # Add a copy of the current stack
            current_stack = [] # Reset for the next stack
        else:
            current_stack.append(fname)

    # Add any remaining files as the last stack if it's not empty
    if current_stack:
        stacks.append(list(current_stack))
    return stacks

def slugify(name):
    """
    Simple slugify function to create safe directory/file names.
    """
    name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    return "".join(c for c in name if c.isalnum() or c == '_').strip('_')

def crop_and_save(image_path, annotations_for_image, output_base_dir, image_name, id_to_category_map, taxon_merge_map):
    """
    Crops objects from a single image based on its annotations and saves them.
    Returns the number of crops successfully saved from this image.
    """
    crops_saved_count = 0
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Warning: Image file not found {image_path}, skipping.")
        return 0
    except Exception as e:
        print(f"Warning: Could not open image {image_path}, error: {e}, skipping.")
        return 0

    for ann_idx, ann in enumerate(annotations_for_image):
        try:
            x, y, w, h = map(int, ann["bbox"])
            if w <= 0 or h <= 0:
                print(f"Warning: Invalid bbox dimensions (w={w}, h={h}) for annotation ID {ann.get('id', 'N/A')} in {image_name}. Skipping crop.")
                continue

            cropped = image.crop((x, y, x + w, y + h))
            label_id = ann["category_id"]
            raw_label_name = id_to_category_map.get(label_id, f"unknown_category_{label_id}")

            # Normalize label from JSON file to handle "Supp" prefixes
            normalized_label_name = normalize_taxon_name(raw_label_name)

            # Apply the taxon merge map to the normalized name
            final_label_name = taxon_merge_map.get(normalized_label_name, normalized_label_name)
            
            label_name_slug = slugify(final_label_name)

            label_dir = os.path.join(output_base_dir, label_name_slug)
            os.makedirs(label_dir, exist_ok=True)

            # Create a unique name for the crop
            crop_name = f"{os.path.splitext(image_name)[0]}_crop{ann.get('id', ann_idx)}.jpg"
            cropped.save(os.path.join(label_dir, crop_name))
            crops_saved_count += 1
        except Exception as e:
            print(f"Error processing annotation {ann.get('id', 'N/A')} for image {image_name}: {e}")

    return crops_saved_count

# Helper functions to save and plot stack counts 
def write_stack_counts_to_csv(stack_counts, output_dir):
    """Writes the taxon stack counts to a CSV file."""
    output_path = os.path.join(output_dir, "taxon_stack_counts.csv")
    print(f"\nWriting stack counts to {output_path}...")
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Taxon', 'NumberOfStacks'])
        for taxon, count in sorted(stack_counts.items()):
            writer.writerow([taxon, count])

def plot_stack_counts(stack_counts, output_dir):
    """Creates and saves a bar chart of taxon stack counts."""
    output_path = os.path.join(output_dir, "taxon_stack_counts.png")
    print(f"Generating stack count plot at {output_path}...")

    sorted_taxa = sorted(stack_counts.keys(), key=lambda k: stack_counts[k], reverse=True)
    sorted_counts = [stack_counts[k] for k in sorted_taxa]

    plt.figure(figsize=(10, max(8, len(sorted_taxa) * 0.4)))
    plt.barh(sorted_taxa, sorted_counts)
    plt.xlabel("Number of Stacks")
    plt.ylabel("Taxon")
    plt.title("Number of Stacks per Taxon")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def write_comprehensive_summary(taxon_stats, output_dir):
    """Writes a comprehensive summary of counts to the console and a CSV file."""
    output_path = os.path.join(output_dir, "comprehensive_taxon_summary.csv")
    print("\nComprehensive Taxon Summary")
    print(f"Writing summary to {output_path}...")

    header = ['Taxon', 'Original Images', 'Stacks', 'Total Cropped Grains']
    rows = []
    for taxon_slug, counts in sorted(taxon_stats.items()):
        rows.append([
            taxon_slug,
            counts['original_image_count'],
            counts['stack_count'],
            counts['cropped_grain_count']
        ])

    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
    except Exception as e:
        print(f"Error writing summary CSV: {e}")

    print(f"{header[0]:<30} | {header[1]:>15} | {header[2]:>8} | {header[3]:>20}")
    print("-" * 80)
    for row in rows:
        print(f"{row[0]:<30} | {row[1]:>15} | {row[2]:>8} | {row[3]:>20}")
    print("---------------------------------")


def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    for split_name in SPLIT_RATIOS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)
    print(f"Created output directory structure in: {OUTPUT_DIR}")
    
    taxon_to_stacks = defaultdict(list)
    taxon_stats = defaultdict(lambda: {'original_image_count': 0, 'stack_count': 0, 'cropped_grain_count': 0})

    print("Collecting image stacks from all taxa...")
    for taxon_folder_name in os.listdir(BASE_DIR):
        taxon_base_path = os.path.join(BASE_DIR, taxon_folder_name)
        if taxon_folder_name == os.path.basename(OUTPUT_DIR) or not os.path.isdir(taxon_base_path):
            continue

        # Normalize the folder name to handle "Supp_" prefixes
        normalized_folder_name = normalize_taxon_name(taxon_folder_name)

        # Apply the merge map to the normalized name
        effective_taxon_name = TAXON_MERGE_MAP.get(normalized_folder_name, normalized_folder_name)
        final_slug = slugify(effective_taxon_name)
        
        if normalized_folder_name != taxon_folder_name or effective_taxon_name != normalized_folder_name:
            print(f"  Processing taxon: {taxon_folder_name} (grouping into -> {final_slug})")
        else:
            print(f"  Processing taxon: {taxon_folder_name}")

        image_files_in_taxon = [f for f in os.listdir(taxon_base_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        annotation_file_path = os.path.join(taxon_base_path, "instances_default.json")
        
        # Track original image count
        taxon_stats[final_slug]['original_image_count'] += len(image_files_in_taxon)

        if not image_files_in_taxon:
            print(f"No image files found in {taxon_folder_name}, skipping.")
            continue
        if not os.path.exists(annotation_file_path):
            print(f"Annotation file 'instances_default.json' not found in {taxon_folder_name}, skipping.")
            continue

        stacks_from_taxon = group_stacks(image_files_in_taxon)
        # Track stack count
        taxon_stats[final_slug]['stack_count'] += len(stacks_from_taxon)

        if not stacks_from_taxon:
            print(f"No stacks formed for taxon {taxon_folder_name}, skipping.")
            continue

        print(f"Found {len(stacks_from_taxon)} stacks for taxon {taxon_folder_name}.")

        for stack in stacks_from_taxon:
            taxon_to_stacks[final_slug].append({
                'files': stack,
                'taxon_name': final_slug,
                'annotation_file': annotation_file_path,
                'taxon_base_path': taxon_base_path
            })

    taxon_to_stacks = {k: v for k, v in taxon_to_stacks.items() if len(v) >= 3}
    print(f"\nTotal valid taxa with >= 3 stacks: {len(taxon_to_stacks)}")

    taxon_stack_counts = {taxon: len(stacks) for taxon, stacks in taxon_to_stacks.items()}
    if taxon_stack_counts:
        write_stack_counts_to_csv(taxon_stack_counts, OUTPUT_DIR)
        plot_stack_counts(taxon_stack_counts, OUTPUT_DIR)

    train_stacks, val_stacks, test_stacks = [], [], []
    from math import floor

    for taxon, stacks in taxon_to_stacks.items():
        random.shuffle(stacks)
        n = len(stacks)
        n_train = max(1, floor(n * SPLIT_RATIOS['train']))
        n_val = max(1, floor(n * SPLIT_RATIOS['val']))
        n_test = n - (n_train + n_val)
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1
        total = n_train + n_val + n_test
        if total < n:
            n_train += (n - total)

        train_stacks.extend(stacks[:n_train])
        val_stacks.extend(stacks[n_train:n_train + n_val])
        test_stacks.extend(stacks[n_train + n_val:n_train + n_val + n_test])
        print(f"Taxon '{taxon}': total {n} -> train: {n_train}, val: {n_val}, test: {n_test}")

    splits_map = {"train": train_stacks, "val": val_stacks, "test": test_stacks}
    
    def count_taxa_in_split(stacks_list):
        return set(stack['taxon_name'] for stack in stacks_list)

    print("\nTaxon Coverage per Split")
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
            try:
                with open(stack_info['annotation_file'], 'r') as f:
                    coco_data = json.load(f)
            except Exception as e:
                print(f"    Error loading annotation file {stack_info['annotation_file']}: {e}. Skipping stack.")
                continue

            id_to_category_map = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
            image_id_map = {os.path.basename(img["file_name"]): img["id"] for img in coco_data.get("images", [])}
            annotations_by_img_id = defaultdict(list)
            for ann in coco_data.get("annotations", []):
                annotations_by_img_id[ann["image_id"]].append(ann)

            for image_filename_in_stack in stack_info['files']:
                image_id = image_id_map.get(image_filename_in_stack)
                if image_id is None:
                    print(f"    Warning: Image {image_filename_in_stack} not found in annotation JSON. Skipping.")
                    continue

                annotations_for_current_image = annotations_by_img_id.get(image_id, [])
                if not annotations_for_current_image:
                    continue

                image_full_path = os.path.join(stack_info['taxon_base_path'], image_filename_in_stack)
                
                # Capture the returned crop count
                num_cropped = crop_and_save(image_full_path, annotations_for_current_image,
                                             output_subdir_for_split, image_filename_in_stack,
                                             id_to_category_map, TAXON_MERGE_MAP)
                
                final_slug = stack_info['taxon_name']
                taxon_stats[final_slug]['cropped_grain_count'] += num_cropped

    print("\nCropping and splitting process complete.")
    write_comprehensive_summary(taxon_stats, OUTPUT_DIR)
    print_dataset_summary(OUTPUT_DIR)

def print_dataset_summary(output_dir):
    print("\nDataset Summary (Cropped Images per Split)")
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
    print("--------------------------------------------------")


if __name__ == "__main__":
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: scikit-learn is required.")
        exit()
    try:
        import matplotlib
    except ImportError:
        print("Error: matplotlib is required.")
        exit()

    if not np.isclose(sum(SPLIT_RATIOS.values()), 1.0):
        print(f"Error: SPLIT_RATIOS must sum to 1.0. Current sum: {sum(SPLIT_RATIOS.values())}")
        exit()

    main()