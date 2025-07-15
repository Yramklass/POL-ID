from pathlib import Path
import shutil

def fix_sp_naming(root_dir):
    """Convert all 'sp_' to 'sp' in folder names within train/val/test subdirectories"""
    root_path = Path(root_dir)
    
    # Process each dataset split
    for split in ['train', 'val', 'test']:
        split_path = root_path / split
        if not split_path.exists():
            print(f"Warning: {split} directory not found, skipping")
            continue
            
        print(f"\nProcessing {split} directory...")
        
        # Find all class folders with 'sp_' in their names
        for old_path in split_path.glob('*sp_*'):
            if old_path.is_dir():
                new_name = old_path.name.replace('sp_', 'sp')
                
                # Only rename if the name actually changes
                if new_name != old_path.name:
                    new_path = old_path.parent / new_name
                    
                    # Handle potential name conflicts
                    if new_path.exists():
                        print(f"Conflict: {new_name} already exists, skipping")
                        continue
                        
                    # Perform the rename
                    old_path.rename(new_path)
                    print(f"Renamed: {old_path.name} → {new_name}")
                    
        print(f"Completed processing {split}")

if __name__ == "__main__":
    processed_crops_folder = "/home/yash/POL-ID/data/classification/processed_crops"
    fix_sp_naming(processed_crops_folder)
    print("\nFolder name standardization complete!")