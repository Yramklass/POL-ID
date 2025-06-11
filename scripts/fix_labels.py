# /scripts/fix_labels.py
from pathlib import Path
import os

# Configuration
try:
    # Assumes this script is in 'scripts/', so its parent's parent is the project root
    ROOT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive environments
    ROOT_DIR = Path(os.getcwd())

# Path to the directory containing 'train', 'val', 'test' label folders
LABELS_BASE_DIR = ROOT_DIR / "data/detection/pollen_detector/datasets/pollen/labels"


def fix_label_files(labels_dir):
    """
    Iterates through all .txt files in a directory and its subdirectories,
    ensuring the class ID (the first number on each line) is set to 0.
    """
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found at '{labels_dir}'")
        return

    print(f"Scanning for label files in '{labels_dir}'...")
    
    # Use rglob to find all .txt files recursively (in train/, val/, test/)
    label_files = list(labels_dir.rglob("*.txt"))
    
    if not label_files:
        print("No label files found to fix.")
        return

    files_changed = 0
    for label_file in label_files:
        corrected_lines = []
        file_needs_correction = False
        
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue # Skip empty lines
            
            # Check if the class ID is already '0'
            if parts[0] != '0':
                file_needs_correction = True
                parts[0] = '0' # Force the class ID to be 0
                
            corrected_lines.append(" ".join(parts))

        # If any line was changed, rewrite the entire file
        if file_needs_correction:
            files_changed += 1
            with open(label_file, 'w') as f:
                f.write("\n".join(corrected_lines))

    print(f"Done. Scanned {len(label_files)} files. Corrected {files_changed} files.")


if __name__ == "__main__":
    fix_label_files(LABELS_BASE_DIR)