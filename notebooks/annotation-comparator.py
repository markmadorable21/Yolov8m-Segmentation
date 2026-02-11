#!/usr/bin/env python3
"""
Script to compare original input images with annotated output images.
Creates a new folder with original images that have corresponding annotations.
Useful for YOLO re-training after running the segmentation pipeline.
"""

import os
import shutil
from pathlib import Path
import glob

class AnnotationComparator:
    def __init__(self, original_input_folder, annotated_output_folder, output_folder):
        """
        Initialize comparator.
        
        Args:
            original_input_folder: Folder with original input images
            annotated_output_folder: Folder with annotated images from pipeline
            output_folder: Where to save matched original images
        """
        self.original_input_folder = Path(original_input_folder)
        self.annotated_output_folder = Path(annotated_output_folder)
        self.output_folder = Path(output_folder)
        
        print("="*70)
        print("ANNOTATION COMPARATOR")
        print("="*70)
        print(f"Original input: {self.original_input_folder}")
        print(f"Annotated output: {self.annotated_output_folder}")
        print(f"Output folder: {self.output_folder}")
        print("="*70)
    
    def get_image_files(self, folder_path):
        """Get all image files from a folder."""
        extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', 
                     '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF',
                     '.webp', '.WEBP']
        
        image_files = []
        for ext in extensions:
            pattern = str(folder_path / f'*{ext}')
            files = glob.glob(pattern)
            image_files.extend([Path(f) for f in files])
        
        return sorted(image_files)
    
    def get_yolo_label_files(self, folder_path):
        """Get all YOLO label files from a folder."""
        txt_files = list(folder_path.glob("*.txt"))
        return sorted(txt_files)
    
    def find_corresponding_original(self, annotated_filename, original_images):
        """
        Find corresponding original image for an annotated image.
        
        Args:
            annotated_filename: Name of annotated image
            original_images: List of original image paths
            
        Returns:
            Path to corresponding original image or None
        """
        # Try exact match first
        for orig_img in original_images:
            if orig_img.name == annotated_filename:
                return orig_img
        
        # Try different extensions
        base_name = Path(annotated_filename).stem
        
        # Common extension variations
        extension_variations = [
            '', '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
            '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF', '.WEBP'
        ]
        
        for ext in extension_variations:
            test_filename = base_name + ext if ext else base_name
            for orig_img in original_images:
                if orig_img.name == test_filename:
                    return orig_img
        
        return None
    
    def copy_matching_images(self, use_labels_for_verification=True):
        """
        Copy original images that have corresponding annotations.
        
        Args:
            use_labels_for_verification: Also check if YOLO label file exists
            
        Returns:
            Dictionary with comparison results
        """
        # Get files from both folders
        original_images = self.get_image_files(self.original_input_folder)
        annotated_images = self.get_image_files(self.annotated_output_folder)
        
        # Also get YOLO labels if available (check in parent directory)
        yolo_labels_folder = self.annotated_output_folder.parent / "yolo_labels"
        yolo_labels = []
        if yolo_labels_folder.exists() and use_labels_for_verification:
            yolo_labels = self.get_yolo_label_files(yolo_labels_folder)
        
        print(f"\nFound {len(original_images)} original images")
        print(f"Found {len(annotated_images)} annotated images")
        if yolo_labels:
            print(f"Found {len(yolo_labels)} YOLO label files")
        
        # Create output directory
        if self.output_folder.exists():
            shutil.rmtree(self.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        stats = {
            'total_original_images': len(original_images),
            'total_annotated_images': len(annotated_images),
            'total_yolo_labels': len(yolo_labels),
            'images_copied': 0,
            'images_with_annotations_only': 0,
            'images_with_labels_only': 0,
            'images_with_both': 0,
            'missing_originals': [],
            'missing_annotations': [],
            'missing_labels': []
        }
        
        print(f"\nComparing and copying matching images...")
        
        # Method 1: Copy based on annotated images
        print(f"\nMethod 1: Based on annotated images")
        for idx, annotated_img in enumerate(annotated_images, 1):
            print(f"[{idx}/{len(annotated_images)}] Checking: {annotated_img.name}")
            
            # Find corresponding original
            original_img = self.find_corresponding_original(annotated_img.name, original_images)
            
            if original_img:
                # Check if YOLO label exists
                base_name = annotated_img.stem
                label_exists = any(label.stem == base_name for label in yolo_labels) if yolo_labels else True
                
                # Copy original image
                output_path = self.output_folder / original_img.name
                shutil.copy2(original_img, output_path)
                stats['images_copied'] += 1
                
                if label_exists:
                    stats['images_with_both'] += 1
                    print(f"  ✓ Copied: {original_img.name} (has annotation & label)")
                else:
                    stats['images_with_annotations_only'] += 1
                    print(f"  ✓ Copied: {original_img.name} (has annotation, no label)")
            else:
                stats['missing_originals'].append(annotated_img.name)
                print(f"  ✗ No original found for: {annotated_img.name}")
        
        
        # Generate summary
        self.generate_summary(stats)
        
        return stats
    
    def generate_summary(self, stats):
        """Generate and display summary report."""
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n📊 Statistics:")
        print(f"  Original images: {stats['total_original_images']}")
        print(f"  Annotated images: {stats['total_annotated_images']}")
        print(f"  YOLO label files: {stats['total_yolo_labels']}")
        print(f"  Images copied: {stats['images_copied']}")
        
        if stats['images_with_both'] > 0:
            print(f"  Images with both annotations & labels: {stats['images_with_both']}")
        if stats['images_with_annotations_only'] > 0:
            print(f"  Images with annotations only: {stats['images_with_annotations_only']}")
        if stats['images_with_labels_only'] > 0:
            print(f"  Images with labels only: {stats['images_with_labels_only']}")
        
        if stats['missing_originals']:
            print(f"\n⚠ Missing original images for {len(stats['missing_originals'])} annotated files:")
            for missing in stats['missing_originals'][:10]:  # Show first 10
                print(f"  - {missing}")
            if len(stats['missing_originals']) > 10:
                print(f"  ... and {len(stats['missing_originals']) - 10} more")
        
        if stats['total_original_images'] > 0 and stats['total_annotated_images'] > 0:
            annotation_rate = (stats['images_copied'] / stats['total_original_images']) * 100
            print(f"\n📈 Annotation rate: {annotation_rate:.1f}% of original images")
        
        print(f"\n✅ Done!")
        print(f"Copied images are in: {self.output_folder}")
        print(f"\n💡 Use these images with YOLO labels from:")
        print(f"   {self.annotated_output_folder.parent}/yolo_labels/")
    
    def create_yolo_dataset_structure(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Create YOLO dataset structure with train/val/test splits.
        
        Args:
            train_ratio: Proportion for training (default: 0.8)
            val_ratio: Proportion for validation (default: 0.1)
            test_ratio: Proportion for testing (default: 0.1)
        """
        import random
        
        # Verify ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            print(f"⚠ Warning: Ratios sum to {total:.3f}, normalizing to 1.0")
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        # Get all copied images
        copied_images = self.get_image_files(self.output_folder)
        
        if not copied_images:
            print("No images found in output folder. Run copy_matching_images() first.")
            return
        
        # Get YOLO labels
        yolo_labels_folder = self.annotated_output_folder.parent / "yolo_labels"
        if not yolo_labels_folder.exists():
            print(f"YOLO labels folder not found: {yolo_labels_folder}")
            return
        
        # Create dataset structure
        dataset_folder = self.output_folder.parent / "yolo_dataset"
        if dataset_folder.exists():
            shutil.rmtree(dataset_folder)
        
        # Create subdirectories
        train_img_dir = dataset_folder / "train" / "images"
        train_lbl_dir = dataset_folder / "train" / "labels"
        val_img_dir = dataset_folder / "val" / "images"
        val_lbl_dir = dataset_folder / "val" / "labels"
        test_img_dir = dataset_folder / "test" / "images"
        test_lbl_dir = dataset_folder / "test" / "labels"
        
        for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, 
                        test_img_dir, test_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Shuffle and split
        image_list = list(copied_images)
        random.shuffle(image_list)
        
        n_total = len(image_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        train_images = image_list[:n_train]
        val_images = image_list[n_train:n_train + n_val]
        test_images = image_list[n_train + n_val:]
        
        print(f"\nCreating YOLO dataset structure...")
        print(f"  Total images: {n_total}")
        print(f"  Train: {n_train} ({train_ratio*100:.1f}%)")
        print(f"  Validation: {n_val} ({val_ratio*100:.1f}%)")
        print(f"  Test: {n_test} ({test_ratio*100:.1f}%)")
        
        # Copy files to respective directories
        def copy_split(images, img_dir, lbl_dir, split_name):
            copied_count = 0
            for img_path in images:
                base_name = img_path.stem
                
                # Copy image
                dest_img_path = img_dir / img_path.name
                shutil.copy2(img_path, dest_img_path)
                
                # Copy corresponding label
                label_src = yolo_labels_folder / f"{base_name}.txt"
                if label_src.exists():
                    dest_label_path = lbl_dir / f"{base_name}.txt"
                    shutil.copy2(label_src, dest_label_path)
                    copied_count += 1
                else:
                    print(f"  ⚠ Missing label for {img_path.name} in {split_name}")
            
            return copied_count
        
        train_copied = copy_split(train_images, train_img_dir, train_lbl_dir, "train")
        val_copied = copy_split(val_images, val_img_dir, val_lbl_dir, "val")
        test_copied = copy_split(test_images, test_img_dir, test_lbl_dir, "test")
        
        # Create dataset.yaml file
        yaml_content = f"""# YOLOv8 Dataset Configuration
path: {dataset_folder.absolute()}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 1

# Class names
names:
  0: eggplant

# Download script/URL (optional)
# download: https://example.com/dataset.zip
"""
        
        yaml_path = dataset_folder / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ YOLO dataset created!")
        print(f"Dataset location: {dataset_folder}")
        print(f"Train images: {train_copied}")
        print(f"Validation images: {val_copied}")
        print(f"Test images: {test_copied}")
        print(f"Dataset config: {yaml_path}")
        print(f"\n💡 To train YOLOv8, use:")
        print(f"   yolo train data={yaml_path} model=yolov8n-seg.pt epochs=100")


def main():
    """Main function with configuration."""
    # ============= CONFIGURATION =============
    CONFIG = {
        'original_input_folder': "/home/saib/ml_project/data/downsample-segment/input",
        'annotated_output_folder': "/home/saib/ml_project/data/downsample-segment/final_output/annotated_images",
        'output_folder': "/home/saib/ml_project/data/downsample-segment/second_yolo_train_final_images",
        'create_yolo_dataset': True,  # Set to False if you only want to copy images
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1
    }
    # ========================================
    
    # Create comparator and run
    comparator = AnnotationComparator(
        original_input_folder=CONFIG['original_input_folder'],
        annotated_output_folder=CONFIG['annotated_output_folder'],
        output_folder=CONFIG['output_folder']
    )
    
    # Copy matching images
    stats = comparator.copy_matching_images(use_labels_for_verification=True)
    
    # Create YOLO dataset structure if requested
    if CONFIG.get('create_yolo_dataset', False) and stats['images_copied'] > 0:
        comparator.create_yolo_dataset_structure(
            train_ratio=CONFIG['train_ratio'],
            val_ratio=CONFIG['val_ratio'],
            test_ratio=CONFIG['test_ratio']
        )


if __name__ == "__main__":
    main()