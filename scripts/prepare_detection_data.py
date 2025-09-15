"""
prepare_detection_data.py

Description:
    Splits detection images into 'train', 'val', and 'test' sets with YOLO folder structure

Usage:
    python prepare_detection_data.py

Inputs:
    - Path of directory containing all detection images and label files (set in script: SOURCE_DIR)
    - Path to target directory for detection dataset - the processed folder will be saved here (set in script: TARGET_ROOT_DIR)
    - Dataset split ratios (set in script: SPLIT_RATIOS)

Outputs:
    - YOLO directory structure with 'train', 'val' and 'test' folders according to specified split
"""

import os
import shutil
import random
from pathlib import Path

# Configuration
# Path to directory containing detection images and label files
SOURCE_DIR ='path/to/directory'
# Path to desired dataset root directory (YOLO dataset will be created here)
TARGET_ROOT_DIR = 'path/to/directory'

# Split Ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def create_yolo_structure(base_path):
    """Creates the required directory structure for YOLO datasets."""
    print(f"Creating directory structure inside '{base_path}'...")
   
    yolo_root = base_path / "datasets" / "pollen"
    for sub in ["images", "labels"]:
        for split in SPLIT_RATIOS.keys():
            (yolo_root / sub / split).mkdir(parents=True, exist_ok=True)
    print("Directory structure created successfully.\n")
    return yolo_root

def find_and_pair_files(source_dir):
    """Finds and pairs all image and label files."""
    file_pairs = []
    print(f"Scanning for projects in '{source_dir.resolve()}'...")

    for project_folder in source_dir.iterdir():
        if not project_folder.is_dir() or project_folder.name == TARGET_ROOT_DIR.name or project_folder.name.startswith('.'):
            continue

        print(f"  Processing folder: '{project_folder.name}'")
        
        try:
            label_dir = next(project_folder.rglob('obj_train_data'))
            print(f"    Found label data in: '{label_dir}'")
        except StopIteration:
            print(f"    WARNING: No 'obj_train_data' folder found in '{project_folder.name}'. Skipping.")
            continue

        image_files = list(project_folder.glob("*.jpg"))
        if not image_files:
            print(f"    WARNING: No .jpg images found in '{project_folder.name}'.")
            continue

        found_pairs_in_folder = 0
        for img_path in image_files:
            label_filename = img_path.with_suffix(".txt").name
            label_path = label_dir / label_filename

            if label_path.exists():
                file_pairs.append((img_path, label_path))
                found_pairs_in_folder += 1
            else:
                print(f"    WARNING: Label for '{img_path.name}' not found. Skipping image.")
        
        print(f"    Found {found_pairs_in_folder} image/label pairs.")

    return file_pairs

def split_and_copy_files(file_pairs, dest_base_path):
    """Shuffles, splits, and copies the file pairs to the new structure."""
    if not file_pairs:
        print("\nNo file pairs found to process. Exiting.")
        return

    print(f"\nFound a total of {len(file_pairs)} image/label pairs.")
    print("Shuffling and splitting the dataset...")

    random.shuffle(file_pairs)

    total = len(file_pairs)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": file_pairs[:train_end],
        "val": file_pairs[train_end:val_end],
        "test": file_pairs[val_end:]
    }

    for split_name, pairs in splits.items():
        count = len(pairs)
        print(f"Copying {count} files to '{split_name}' set...")
        if count == 0:
            continue
            
        img_dest_dir = dest_base_path / "images" / split_name
        lbl_dest_dir = dest_base_path / "labels" / split_name

        for img_src, lbl_src in pairs:
            shutil.copy2(img_src, img_dest_dir)
            shutil.copy2(lbl_src, lbl_dest_dir)

def main():
    """Main function for data preparation."""
    if TARGET_ROOT_DIR.exists():
        print(f"Target directory '{TARGET_ROOT_DIR}' already exists.")
        print("Please remove or rename it if you want to run a fresh preparation.")
        return

    yolo_base_path = create_yolo_structure(TARGET_ROOT_DIR)
    all_file_pairs = find_and_pair_files(SOURCE_DIR)
    split_and_copy_files(all_file_pairs, yolo_base_path)

    print("Finished.")
    print(f"Clean dataset is ready in the '{TARGET_ROOT_DIR}' folder.")


if __name__ == "__main__":
    main()