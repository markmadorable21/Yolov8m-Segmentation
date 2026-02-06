import splitfolders
import os
import shutil
from pathlib import Path

def split_yolo_dataset_robust(input_dir, output_dir, ratio=(0.8, 0.2), seed=42):
    """
    Split YOLO dataset with robust handling for mismatched images/labels.
    
    Args:
        input_dir: Directory with 'images' and 'labels' subfolders
        output_dir: Output directory for split dataset
        ratio: Train/test ratio or train/val/test ratio
        seed: Random seed for reproducibility
    """
    
    print("=" * 60)
    print("YOLO DATASET SPLITTER")
    print("=" * 60)
    
    # Paths
    images_dir = Path(input_dir) / "images"
    labels_dir = Path(input_dir) / "labels"
    
    # Check if directories exist
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Get all files
    image_files = list(images_dir.glob("*.*"))
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"\nFound in images/: {len(image_files)} files")
    print(f"Found in labels/: {len(label_files)} files")
    
    # Create dictionary of base names
    image_basenames = {f.stem: f for f in image_files}
    label_basenames = {f.stem: f for f in label_files}
    
    # Find common basenames (pairs that exist in both)
    common_basenames = set(image_basenames.keys()) & set(label_basenames.keys())
    
    # Find orphaned files
    images_only = set(image_basenames.keys()) - common_basenames
    labels_only = set(label_basenames.keys()) - common_basenames
    
    print(f"\nMatching pairs found: {len(common_basenames)}")
    print(f"Images without labels: {len(images_only)}")
    print(f"Labels without images: {len(labels_only)}")
    
    # Show orphaned files (optional)
    if images_only:
        print("\nOrphaned images (will be ignored):")
        for name in sorted(list(images_only))[:10]:  # Show first 10
            print(f"  - {name}")
        if len(images_only) > 10:
            print(f"  ... and {len(images_only) - 10} more")
    
    if labels_only:
        print("\nOrphaned labels (will be ignored):")
        for name in sorted(list(labels_only))[:10]:  # Show first 10
            print(f"  - {name}")
        if len(labels_only) > 10:
            print(f"  ... and {len(labels_only) - 10} more")
    
    # Create temporary directory with only matched pairs
    temp_dir = Path("temp_split_yolo")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\nCopying {len(common_basenames)} matched pairs to temporary directory...")
    
    # Copy matched pairs to temp directory
    matched_count = 0
    for basename in common_basenames:
        # Source paths
        src_img = image_basenames[basename]
        src_label = label_basenames[basename]
        
        # Get image extension
        img_ext = src_img.suffix
        
        # Destination paths in temp directory
        # We'll use .jpg extension for all images to make split-folders happy
        dst_img = temp_dir / f"{basename}.jpg"
        dst_label = temp_dir / f"{basename}.jpg.txt"  # Special naming
        
        # Copy image (convert to .jpg name)
        shutil.copy2(src_img, dst_img)
        
        # Copy label file with special naming
        shutil.copy2(src_label, dst_label)
        
        matched_count += 1
    
    print(f"Copied {matched_count} image-label pairs")
    
    # Run split-folders
    print(f"\nSplitting with ratio {ratio}...")
    
    try:
        splitfolders.ratio(
            str(temp_dir), 
            output=str(output_dir),
            seed=seed,
            ratio=ratio,  # (train, test) or (train, val, test)
            group_prefix=None,
            move=False,  # Copy instead of move
        )
    except Exception as e:
        print(f"Error during split: {e}")
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    
    # Reorganize the split folders to YOLO structure
    print("\nReorganizing to YOLO structure...")
    reorganize_split_folders_robust(output_dir)
    
    # Clean up temporary directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Print final statistics
    print_final_stats(output_dir)
    
    # Create data.yaml automatically
    create_data_yaml_auto(output_dir, ratio)
    
    print(f"\n" + "=" * 60)
    print("SPLITTING COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"data.yaml created at: {output_dir}/data.yaml")

def reorganize_split_folders_robust(output_dir):
    """
    Reorganize split-folders output to proper YOLO structure.
    Handles various image extensions.
    """
    output_path = Path(output_dir)
    
    # Define possible splits
    possible_splits = ['train', 'val', 'test', 'training', 'validation', 'testing']
    
    for split_dir in output_path.iterdir():
        if split_dir.is_dir() and split_dir.name.lower() in [s.lower() for s in possible_splits]:
            split_name = split_dir.name.lower()
            
            # Standardize split name
            if split_name == 'training':
                split_name = 'train'
            elif split_name == 'validation':
                split_name = 'val'
            elif split_name == 'testing':
                split_name = 'test'
            
            # Create new directory structure
            new_split_dir = output_path / split_name
            images_dir = new_split_dir / 'images'
            labels_dir = new_split_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Process files in the split directory
            files_moved = 0
            for file_path in split_dir.iterdir():
                if file_path.is_file():
                    filename = file_path.name
                    
                    if filename.endswith('.jpg.txt'):
                        # This is a label file
                        # Extract original base name
                        base_name = filename[:-8]  # Remove '.jpg.txt'
                        
                        # Check if we have the corresponding image
                        possible_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                        image_found = False
                        
                        for ext in possible_image_extensions:
                            possible_image = split_dir / f"{base_name}{ext}"
                            if possible_image.exists():
                                # Move image with original extension
                                shutil.move(str(possible_image), str(images_dir / f"{base_name}{ext}"))
                                # Move label with .txt extension
                                shutil.move(str(file_path), str(labels_dir / f"{base_name}.txt"))
                                image_found = True
                                files_moved += 2
                                break
                        
                        if not image_found:
                            print(f"Warning: No matching image found for label {filename}")
                    elif any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']):
                        # This is an image file without special naming
                        # Check if it has a corresponding label
                        base_name = file_path.stem
                        possible_label = split_dir / f"{base_name}.jpg.txt"
                        
                        if possible_label.exists():
                            # Move both
                            shutil.move(str(file_path), str(images_dir / filename))
                            shutil.move(str(possible_label), str(labels_dir / f"{base_name}.txt"))
                            files_moved += 2
                        else:
                            # Check for regular .txt label
                            regular_label = split_dir / f"{base_name}.txt"
                            if regular_label.exists():
                                shutil.move(str(file_path), str(images_dir / filename))
                                shutil.move(str(regular_label), str(labels_dir / f"{base_name}.txt"))
                                files_moved += 2
                            else:
                                print(f"Warning: No label found for image {filename}")
            
            print(f"  {split_name}: Moved {files_moved//2} image-label pairs")
            
            # Remove old directory if empty
            if split_dir.exists() and split_dir != new_split_dir:
                try:
                    split_dir.rmdir()
                except:
                    pass

def print_final_stats(output_dir):
    """Print statistics of the split dataset."""
    output_path = Path(output_dir)
    
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    
    total_images = 0
    total_labels = 0
    
    for split_dir in output_path.iterdir():
        if split_dir.is_dir() and split_dir.name in ['train', 'val', 'test']:
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                num_images = len(list(images_dir.glob("*.*")))
                num_labels = len(list(labels_dir.glob("*.txt")))
                
                print(f"{split_dir.name.upper():10s}: {num_images:4d} images, {num_labels:4d} labels")
                
                # Check for mismatches
                if num_images != num_labels:
                    print(f"  WARNING: Mismatch in {split_dir.name}! ({num_images} images vs {num_labels} labels)")
                    
                    # List mismatches
                    image_basenames = {f.stem for f in images_dir.glob("*.*")}
                    label_basenames = {f.stem for f in labels_dir.glob("*.txt")}
                    
                    missing_labels = image_basenames - label_basenames
                    missing_images = label_basenames - image_basenames
                    
                    if missing_labels:
                        print(f"    Images without labels: {len(missing_labels)}")
                    if missing_images:
                        print(f"    Labels without images: {len(missing_images)}")
                
                total_images += num_images
                total_labels += num_labels
    
    print("-" * 60)
    print(f"TOTAL:       {total_images:4d} images, {total_labels:4d} labels")
    
    if total_images != total_labels:
        print(f"\nCRITICAL WARNING: Total images ({total_images}) != total labels ({total_labels})")
        print("Some pairs may be missing!")
    else:
        print(f"\nPerfect! All {total_images} images have matching labels.")

def create_data_yaml_auto(output_dir, ratio):
    """Automatically create data.yaml configuration file."""
    output_path = Path(output_dir)
    
    # Check which splits exist
    splits = []
    for split_name in ['train', 'val', 'test']:
        split_dir = output_path / split_name / 'images'
        if split_dir.exists() and any(split_dir.iterdir()):
            splits.append(split_name)
    
    if not splits:
        raise ValueError("No valid splits found in output directory")
    
    # Build YAML content
    yaml_lines = [
        "# YOLOv8 Dataset Configuration",
        f"path: {output_path.absolute()}",
        ""
    ]
    
    # Add split paths
    for split in splits:
        yaml_lines.append(f"{split}: {split}/images")
    
    yaml_lines.extend([
        "",
        "# Number of classes",
        "nc: 1  # UPDATE WITH YOUR NUMBER OF CLASSES",
        "",
        "# Class names",
        "names: ['eggplant']  # UPDATE WITH YOUR CLASS NAMES",
        "",
        "# Dataset information",
        f"split_ratio: {ratio}",
        f"total_images: {sum([len(list((output_path / split / 'images').glob('*.*'))) for split in splits])}",
        "description: Eggplant segmentation dataset",
        "created_with: split_yolo_dataset_robust.py",
        "date: auto-generated"
    ])
    
    # Write to file
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write("\n".join(yaml_lines))
    
    return yaml_path

def cleanup_orphaned_files(input_dir, move_to_folder="orphaned_files"):
    """
    Optional: Move orphaned files to separate folder for inspection.
    """
    input_path = Path(input_dir)
    orphan_dir = input_path.parent / move_to_folder
    
    images_dir = input_path / "images"
    labels_dir = input_path / "labels"
    
    # Find all files
    image_files = list(images_dir.glob("*.*"))
    label_files = list(labels_dir.glob("*.txt"))
    
    # Create sets of basenames
    image_basenames = {f.stem: f for f in image_files}
    label_basenames = {f.stem: f for f in label_files}
    
    # Find orphans
    images_only = set(image_basenames.keys()) - set(label_basenames.keys())
    labels_only = set(label_basenames.keys()) - set(image_basenames.keys())
    
    if not images_only and not labels_only:
        print("No orphaned files found.")
        return
    
    # Create orphan directory
    orphan_dir.mkdir(exist_ok=True)
    (orphan_dir / "images_without_labels").mkdir(exist_ok=True)
    (orphan_dir / "labels_without_images").mkdir(exist_ok=True)
    
    # Move orphaned images
    for basename in images_only:
        src = image_basenames[basename]
        dst = orphan_dir / "images_without_labels" / src.name
        shutil.move(str(src), str(dst))
        print(f"Moved orphaned image: {src.name}")
    
    # Move orphaned labels
    for basename in labels_only:
        src = label_basenames[basename]
        dst = orphan_dir / "labels_without_images" / src.name
        shutil.move(str(src), str(dst))
        print(f"Moved orphaned label: {src.name}")
    
    print(f"\nMoved {len(images_only)} orphaned images and {len(labels_only)} orphaned labels to {orphan_dir}")

# Usage with example
if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/home/saib/ml_project/data/training_yolo_dataset"  # Your folder with images/ and labels/
    OUTPUT_DIR = "yolo_dataset_split"  # Output folder
    
    # Split ratios
    # For train/test only: (0.8, 0.2)
    # For train/val/test: (0.7, 0.15, 0.15)
    SPLIT_RATIO = (0.8, 0.2)  # 80% train, 20% test
    
    # Optional: Clean up orphaned files first
    # cleanup_orphaned_files(INPUT_DIR)
    
    # Run the split
    split_yolo_dataset_robust(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        ratio=SPLIT_RATIO,
        seed=42  # For reproducibility
    )
    
    print(f"\nNext steps:")
    print(f"1. Check the output in: {OUTPUT_DIR}")
    print(f"2. Verify data.yaml file")
    print(f"3. Train YOLO with: yolo segment train data={OUTPUT_DIR}/data.yaml model=yolov8n-seg.pt")