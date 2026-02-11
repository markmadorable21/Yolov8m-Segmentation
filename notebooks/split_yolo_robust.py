import splitfolders
import os
import shutil
from pathlib import Path
import sys

def split_yolo_dataset_robust(input_dir, output_dir, ratio=(0.7, 0.3), seed=42, batch_size=500):
    """
    Split YOLO dataset with robust handling for mismatched images/labels.
    Uses batch processing to avoid memory issues.
    
    Args:
        input_dir: Directory with 'images' and 'labels' subfolders
        output_dir: Output directory for split dataset
        ratio: Train/test ratio or train/val/test ratio
        seed: Random seed for reproducibility
        batch_size: Number of files to process in each batch (default: 500)
    """
    
    print("=" * 60)
    print("YOLO DATASET SPLITTER (Memory-Efficient)")
    print("=" * 60)
    
    # Paths
    images_dir = Path(input_dir) / "second_yolo_train_final_images"
    labels_dir = Path(input_dir) / "second_yolo_train_final_labels"
    
    # Check if directories exist
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Get all files in a memory-efficient way
    print(f"\nScanning directories...")
    
    # Use generator to find image files in batches
    def get_image_files_batch():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
                           '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF', '.WEBP']
        for ext in image_extensions:
            pattern = str(images_dir / f'*{ext}')
            for img_path in glob.iglob(pattern):
                yield Path(img_path)
    
    # Get label files in a memory-efficient way
    def get_label_files_batch():
        pattern = str(labels_dir / '*.txt')
        for lbl_path in glob.iglob(pattern):
            yield Path(lbl_path)
    
    # Count files first
    print("Counting files...")
    from itertools import islice
    import glob
    
    # Count images
    image_count = 0
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']:
        pattern = str(images_dir / f'*{ext}')
        image_count += len(glob.glob(pattern))
    
    # Count labels
    label_count = len(list(labels_dir.glob("*.txt")))
    
    print(f"Found in images/: {image_count} files")
    print(f"Found in labels/: {label_count} files")
    
    # Build index of label stems first (more efficient)
    print("Building label index...")
    label_stems = set()
    label_batch = []
    
    for lbl_path in labels_dir.glob("*.txt"):
        label_stems.add(lbl_path.stem)
        label_batch.append(lbl_path)
        if len(label_batch) % 1000 == 0:
            print(f"  Indexed {len(label_batch)} labels...")
    
    print(f"Indexed {len(label_stems)} unique label stems")
    
    # Find common basenames in batches
    print("\nFinding matching pairs...")
    common_basenames = []
    images_only = []
    
    # Process images in batches
    batch_num = 0
    total_processed = 0
    
    # Get all image files first (but we'll process in batches)
    all_image_files = list(images_dir.glob("*.*"))
    
    for i in range(0, len(all_image_files), batch_size):
        batch_num += 1
        batch = all_image_files[i:i + batch_size]
        
        print(f"Processing batch {batch_num} ({len(batch)} images)...")
        
        for img_path in batch:
            total_processed += 1
            if total_processed % 1000 == 0:
                print(f"  Processed {total_processed}/{image_count} images...")
            
            stem = img_path.stem
            if stem in label_stems:
                common_basenames.append(stem)
            else:
                images_only.append(stem)
    
    labels_only = list(label_stems - set(common_basenames))
    
    print(f"\nMatching pairs found: {len(common_basenames)}")
    print(f"Images without labels: {len(images_only)}")
    print(f"Labels without images: {len(labels_only)}")
    
    # Show orphaned files (optional)
    if images_only:
        print(f"\nOrphaned images (will be ignored): {len(images_only)} files")
        if len(images_only) <= 10:
            for name in sorted(images_only):
                print(f"  - {name}")
    
    if labels_only:
        print(f"\nOrphaned labels (will be ignored):")
        for i, name in enumerate(sorted(labels_only)[:10]):  # Show first 10
            print(f"  - {name}")
        if len(labels_only) > 10:
            print(f"  ... and {len(labels_only) - 10} more")
    
    # Create temporary directory with only matched pairs
    temp_dir = Path("temp_split_yolo")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\nCopying {len(common_basenames)} matched pairs to temporary directory...")
    
    # Copy matched pairs in batches
    matched_count = 0
    batch_num = 0
    
    # Group by first letter for better organization
    basename_dict = {}
    for basename in common_basenames:
        first_char = basename[0].lower() if basename else '_'
        if first_char not in basename_dict:
            basename_dict[first_char] = []
        basename_dict[first_char].append(basename)
    
    # Process each group
    for first_char, basenames in basename_dict.items():
        print(f"  Processing group '{first_char}' ({len(basenames)} files)...")
        
        for basename in basenames:
            try:
                # Find image file with any extension
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']:
                    test_path = images_dir / f"{basename}{ext}"
                    if test_path.exists():
                        img_path = test_path
                        break
                    # Try uppercase extension
                    test_path = images_dir / f"{basename}{ext.upper()}"
                    if test_path.exists():
                        img_path = test_path
                        break
                
                if not img_path:
                    # Try with any extension
                    for file in images_dir.glob(f"{basename}.*"):
                        img_path = file
                        break
                
                if not img_path:
                    print(f"    Warning: Image not found for {basename}")
                    continue
                
                # Find label file
                label_path = labels_dir / f"{basename}.txt"
                if not label_path.exists():
                    print(f"    Warning: Label not found for {basename}")
                    continue
                
                # Destination paths in temp directory
                # Use consistent .jpg extension for split-folders compatibility
                dst_img = temp_dir / f"{basename}.jpg"
                dst_label = temp_dir / f"{basename}.jpg.txt"
                
                # Copy image
                shutil.copy2(img_path, dst_img)
                
                # Copy label
                shutil.copy2(label_path, dst_label)
                
                matched_count += 1
                
                if matched_count % 500 == 0:
                    print(f"    Copied {matched_count} pairs...")
                    
            except Exception as e:
                print(f"    Error copying {basename}: {e}")
                continue
    
    print(f"\nSuccessfully copied {matched_count} image-label pairs")
    
    if matched_count == 0:
        print("❌ No files were copied successfully!")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return
    
    # Check if we have enough files for splitting
    if matched_count < 10:
        print(f"⚠ Warning: Only {matched_count} files available. Consider using all for training.")
    
    # Run split-folders
    print(f"\nSplitting {matched_count} files with ratio {ratio}...")
    
    try:
        splitfolders.ratio(
            str(temp_dir), 
            output=str(output_dir),
            seed=seed,
            ratio=ratio,
            group_prefix=None,
            move=False,
        )
        print("Split completed successfully!")
    except Exception as e:
        print(f"❌ Error during split: {e}")
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    
    # Reorganize the split folders to YOLO structure
    print("\nReorganizing to YOLO structure...")
    reorganize_split_folders_robust(output_dir)
    
    # Clean up temporary directory
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary directory")
        except:
            print("Warning: Could not clean up temporary directory")
    
    # Print final statistics
    print_final_stats(output_dir)
    
    # Create data.yaml automatically
    create_data_yaml_auto(output_dir, ratio)
    
    print(f"\n" + "=" * 60)
    print("SPLITTING COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"data.yaml created at: {output_dir}/data.yaml")
    
    return matched_count

def reorganize_split_folders_robust(output_dir):
    """
    Reorganize split-folders output to proper YOLO structure.
    Handles various image extensions with memory efficiency.
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
            
            # Process files in batches
            files_processed = list(split_dir.iterdir())
            files_moved = 0
            
            print(f"  Processing {split_name} ({len(files_processed)} files)...")
            
            # First pass: Handle .jpg.txt files
            for file_path in files_processed:
                if file_path.is_file() and file_path.name.endswith('.jpg.txt'):
                    base_name = file_path.name[:-8]  # Remove '.jpg.txt'
                    
                    # Find corresponding image
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                        img_file = split_dir / f"{base_name}{ext}"
                        if img_file.exists():
                            # Move both files
                            shutil.move(str(img_file), str(images_dir / f"{base_name}{ext}"))
                            # Rename label to .txt
                            shutil.move(str(file_path), str(labels_dir / f"{base_name}.txt"))
                            files_moved += 2
                            break
            
            # Clean up old directory if empty
            if split_dir.exists() and split_dir != new_split_dir:
                try:
                    split_dir.rmdir()
                except:
                    # Directory not empty, leave it
                    pass
            
            print(f"    Moved {files_moved//2} pairs to {split_name}/")

def print_final_stats(output_dir):
    """Print statistics of the split dataset."""
    output_path = Path(output_dir)
    
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    
    total_images = 0
    total_labels = 0
    has_mismatch = False
    
    for split_name in ['train', 'val', 'test']:
        split_dir = output_path / split_name
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if images_dir.exists() and labels_dir.exists():
            # Count files efficiently
            num_images = 0
            num_labels = 0
            
            # Count images
            for _ in images_dir.glob("*.*"):
                num_images += 1
            
            # Count labels
            for _ in labels_dir.glob("*.txt"):
                num_labels += 1
            
            print(f"{split_name.upper():10s}: {num_images:4d} images, {num_labels:4d} labels")
            
            if num_images != num_labels:
                has_mismatch = True
                print(f"  ⚠ Mismatch: {num_images} images vs {num_labels} labels")
            
            total_images += num_images
            total_labels += num_labels
        elif split_dir.exists():
            print(f"{split_name.upper():10s}: 0 files (directory exists but empty)")
    
    print("-" * 60)
    print(f"TOTAL:       {total_images:4d} images, {total_labels:4d} labels")
    
    if total_images != total_labels:
        print(f"\n⚠ WARNING: Total images ({total_images}) != total labels ({total_labels})")
        print("Some pairs may be incomplete.")
    elif has_mismatch:
        print(f"\n⚠ WARNING: Individual splits have mismatches, but totals match.")
    else:
        print(f"\n✓ Perfect! All {total_images} images have matching labels.")

def create_data_yaml_auto(output_dir, ratio):
    """Automatically create data.yaml configuration file."""
    output_path = Path(output_dir)
    
    # Check which splits exist and have files
    splits = []
    for split_name in ['train', 'val', 'test']:
        split_dir = output_path / split_name / 'images'
        if split_dir.exists():
            # Check if directory has files
            has_files = False
            for _ in split_dir.glob("*.*"):
                has_files = True
                break
            if has_files:
                splits.append(split_name)
    
    if not splits:
        # Check if we have any data at all
        for split_dir in output_path.iterdir():
            if split_dir.is_dir():
                images_dir = split_dir / 'images'
                if images_dir.exists():
                    for _ in images_dir.glob("*.*"):
                        splits.append(split_dir.name)
                        break
    
    if not splits:
        print("⚠ Warning: No data found in output directory")
        splits = ['train', 'val']  # Default
    
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
        "nc: 1",
        "",
        "# Class names",
        "names: ['eggplant']",
        "",
        "# Dataset information",
        f"split_ratio: {ratio}",
        "description: Eggplant segmentation dataset",
        "created_with: split_yolo_dataset_robust.py",
        "date: auto-generated"
    ])
    
    # Write to file
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write("\n".join(yaml_lines))
    
    print(f"Created data.yaml with splits: {', '.join(splits)}")
    return yaml_path

def cleanup_orphaned_files(input_dir, move_to_folder="orphaned_files"):
    """
    Optional: Move orphaned files to separate folder for inspection.
    Uses batch processing for memory efficiency.
    """
    input_path = Path(input_dir)
    orphan_dir = input_path.parent / move_to_folder
    
    images_dir = input_path / "second_yolo_train_final_images"
    labels_dir = input_path / "second_yolo_train_final_labels"
    
    # Find all files efficiently
    print("Scanning for orphaned files...")
    
    # Get image stems
    image_stems = set()
    for img_path in images_dir.glob("*.*"):
        image_stems.add(img_path.stem)
    
    # Get label stems
    label_stems = set()
    for lbl_path in labels_dir.glob("*.txt"):
        label_stems.add(lbl_path.stem)
    
    # Find orphans
    images_only = image_stems - label_stems
    labels_only = label_stems - image_stems
    
    if not images_only and not labels_only:
        print("No orphaned files found.")
        return
    
    # Create orphan directory
    orphan_dir.mkdir(exist_ok=True)
    (orphan_dir / "images_without_labels").mkdir(exist_ok=True)
    (orphan_dir / "labels_without_images").mkdir(exist_ok=True)
    
    # Move orphaned images
    moved_count = 0
    for stem in images_only:
        # Find the image file
        for img_path in images_dir.glob(f"{stem}.*"):
            dst = orphan_dir / "images_without_labels" / img_path.name
            shutil.move(str(img_path), str(dst))
            moved_count += 1
            if moved_count % 100 == 0:
                print(f"Moved {moved_count} orphaned files...")
            break
    
    # Move orphaned labels
    for stem in labels_only:
        lbl_path = labels_dir / f"{stem}.txt"
        if lbl_path.exists():
            dst = orphan_dir / "labels_without_images" / lbl_path.name
            shutil.move(str(lbl_path), str(dst))
            moved_count += 1
            if moved_count % 100 == 0:
                print(f"Moved {moved_count} orphaned files...")
    
    print(f"\nMoved {len(images_only)} orphaned images and {len(labels_only)} orphaned labels to {orphan_dir}")

# Alternative minimal version for very large datasets
def split_yolo_dataset_minimal(input_dir, output_dir, ratio=(0.8, 0.2), seed=42):
    """
    Minimal version that processes files one by one to avoid memory issues.
    """
    print("Using minimal splitter (processes files one by one)...")
    
    images_dir = Path(input_dir) / "second_yolo_train_final_images"
    labels_dir = Path(input_dir) / "second_yolo_train_final_labels"
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process images one by one
    import random
    random.seed(seed)
    
    matched_count = 0
    total_processed = 0
    
    # Process all image files
    for img_path in images_dir.glob("*.*"):
        total_processed += 1
        if total_processed % 1000 == 0:
            print(f"Processed {total_processed} images, matched {matched_count}...")
        
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        
        if label_path.exists():
            matched_count += 1
            
            # Random split based on ratio
            r = random.random()
            if len(ratio) == 2:
                # train/test
                split = 'train' if r < ratio[0] else 'val'
            else:
                # train/val/test
                if r < ratio[0]:
                    split = 'train'
                elif r < ratio[0] + ratio[1]:
                    split = 'val'
                else:
                    split = 'test'
            
            # Copy files
            shutil.copy2(img_path, output_dir / split / 'images' / img_path.name)
            shutil.copy2(label_path, output_dir / split / 'labels' / f"{stem}.txt")
    
    print(f"\nMinimal splitter completed!")
    print(f"Processed {total_processed} images, matched {matched_count} pairs")
    
    # Create data.yaml
    create_data_yaml_auto(output_dir, ratio)
    
    return matched_count

# Usage with example
if __name__ == "__main__":
    try:
        # Configuration
        INPUT_DIR = "/home/saib/ml_project/data/downsample-segment/second_yolov8_training"
        OUTPUT_DIR = "/home/saib/ml_project/data/downsample-segment/second_yolov8_training/outputs"
        
        # Split ratios
        SPLIT_RATIO = (0.7, 0.3)  # 70% train, 30% test
        
        # Choose batch size based on available memory
        # Start with smaller batch size, increase if you have more RAM
        BATCH_SIZE = 500
        
        print(f"Using batch size: {BATCH_SIZE}")
        print(f"Input directory: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Split ratio: {SPLIT_RATIO}")
        
        # Optional: Clean up orphaned files first
        # cleanup_orphaned_files(INPUT_DIR)
        
        # Run the split
        matched_count = split_yolo_dataset_robust(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            ratio=SPLIT_RATIO,
            seed=42,
            batch_size=BATCH_SIZE
        )
        
        if matched_count == 0:
            print("\nTrying minimal splitter instead...")
            # Use minimal version as fallback
            split_yolo_dataset_minimal(
                input_dir=INPUT_DIR,
                output_dir=Path(OUTPUT_DIR) / "minimal_output",
                ratio=SPLIT_RATIO,
                seed=42
            )
        
        print(f"\nNext steps:")
        print(f"1. Check the output in: {OUTPUT_DIR}")
        print(f"2. Verify data.yaml file")
        print(f"3. Train YOLO with: yolo segment train data={OUTPUT_DIR}/data.yaml model=yolov8n-seg.pt")
        
    except MemoryError:
        print("\n❌ Memory error detected!")
        print("Try using the minimal splitter or reducing batch size.")
        print("\nTo use minimal splitter:")
        print(f"  split_yolo_dataset_minimal('{INPUT_DIR}', '{OUTPUT_DIR}', {SPLIT_RATIO})")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()