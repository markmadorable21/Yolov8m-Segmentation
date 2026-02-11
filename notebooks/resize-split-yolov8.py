#!/usr/bin/env python3
"""
YOLO Dataset Preprocessor & Splitter
1. Resizes large images for YOLO training
2. Splits into train/val sets
3. Uses streaming processing to avoid memory issues
"""

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
import sys

class YOLODatasetProcessor:
    def __init__(self, input_dir, output_dir, 
                 max_size_mb=1.0, 
                 target_size=(640, 640),
                 train_ratio=0.7,
                 seed=42):
        """
        Initialize dataset processor.
        
        Args:
            input_dir: Directory with images/ and labels/
            output_dir: Where to save processed dataset
            max_size_mb: Maximum file size in MB (default: 1.0)
            target_size: Target image dimensions (default: 640x640)
            train_ratio: Training set ratio (default: 0.7)
            seed: Random seed
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_size_mb = max_size_mb
        self.target_width, self.target_height = target_size
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        
        # Input directories
        self.images_dir = self.input_dir / "second_yolo_train_final_images"
        self.labels_dir = self.input_dir / "second_yolo_train_final_labels"
        
        # Output directories
        self.train_img_dir = self.output_dir / "train" / "images"
        self.train_lbl_dir = self.output_dir / "train" / "labels"
        self.val_img_dir = self.output_dir / "val" / "images"
        self.val_lbl_dir = self.output_dir / "val" / "labels"
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'matching_pairs': 0,
            'resized_images': 0,
            'copied_images': 0,
            'train_count': 0,
            'val_count': 0,
            'errors': 0
        }
        
        print("=" * 70)
        print("YOLO DATASET PROCESSOR & SPLITTER")
        print("=" * 70)
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Max size: {max_size_mb} MB")
        print(f"Target size: {target_size}")
        print(f"Train ratio: {train_ratio}")
        print("=" * 70)
    
    def validate_directories(self):
        """Check if input directories exist."""
        if not self.images_dir.exists():
            raise ValueError(f"❌ Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"❌ Labels directory not found: {self.labels_dir}")
        
        print("✓ Input directories validated")
        return True
    
    def get_file_size_mb(self, file_path):
        """Get file size in megabytes."""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def resize_image(self, image_path, output_path):
        """
        Resize image to target dimensions while maintaining aspect ratio.
        Also reduces quality if file is still too large.
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  ⚠ Cannot read image: {image_path.name}")
                return False
            
            original_height, original_width = img.shape[:2]
            original_size_mb = self.get_file_size_mb(image_path)
            
            # Calculate scaling factor
            scale_w = self.target_width / original_width
            scale_h = self.target_height / original_height
            scale = min(scale_w, scale_h)
            
            # Don't enlarge small images
            if scale > 1.0:
                scale = 1.0
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Ensure dimensions are divisible by 32 for YOLO
            new_width = (new_width // 32) * 32
            new_height = (new_height // 32) * 32
            
            # Ensure minimum size
            new_width = max(new_width, 320)
            new_height = max(new_height, 320)
            
            # Resize image
            if new_width != original_width or new_height != original_height:
                resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.stats['resized_images'] += 1
            else:
                resized = img
            
            # Determine compression parameters
            quality = 95  # Start with high quality
            
            # If original is very large, reduce quality
            if original_size_mb > 5.0:
                quality = 70
            elif original_size_mb > 2.0:
                quality = 85
            elif original_size_mb > 1.0:
                quality = 90
            
            # Save image with appropriate compression
            ext = image_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), resized, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif ext in ['.png']:
                # PNG compression level (0-9, higher = more compression)
                compression = 3 if original_size_mb > 1.0 else 1
                cv2.imwrite(str(output_path), resized, 
                           [cv2.IMWRITE_PNG_COMPRESSION, compression])
            else:
                cv2.imwrite(str(output_path), resized)
            
            # Check final size
            final_size_mb = self.get_file_size_mb(output_path)
            
            if final_size_mb > self.max_size_mb:
                # Try again with lower quality
                if ext in ['.jpg', '.jpeg']:
                    cv2.imwrite(str(output_path), resized, 
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
                elif ext in ['.png']:
                    cv2.imwrite(str(output_path), resized, 
                               [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error resizing {image_path.name}: {e}")
            self.stats['errors'] += 1
            return False
    
    def copy_label_file(self, label_path, output_path):
        """Copy label file as-is."""
        try:
            shutil.copy2(label_path, output_path)
            return True
        except Exception as e:
            print(f"  ❌ Error copying label {label_path.name}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_dataset(self):
        """
        Process entire dataset: resize images and split into train/val.
        Uses streaming to avoid memory issues.
        """
        # Step 1: Validate
        self.validate_directories()
        
        # Step 2: Create output directories
        self.train_img_dir.mkdir(parents=True, exist_ok=True)
        self.train_lbl_dir.mkdir(parents=True, exist_ok=True)
        self.val_img_dir.mkdir(parents=True, exist_ok=True)
        self.val_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Count total images (for progress tracking)
        print("\n📊 Counting files...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        total_images = 0
        for ext in image_extensions:
            total_images += len(list(self.images_dir.glob(f"*{ext}")))
            total_images += len(list(self.images_dir.glob(f"*{ext.upper()}")))
        
        total_labels = len(list(self.labels_dir.glob("*.txt")))
        
        print(f"  Found {total_images} images")
        print(f"  Found {total_labels} labels")
        
        if total_images == 0:
            print("❌ No images found!")
            return False
        
        # Step 4: Process images
        print(f"\n🔄 Processing {total_images} images...")
        processed_count = 0
        
        for ext in image_extensions:
            # Process both lowercase and uppercase extensions
            for image_path in self.images_dir.glob(f"*{ext}"):
                self.process_single_image(image_path, processed_count, total_images)
                processed_count += 1
            
            for image_path in self.images_dir.glob(f"*{ext.upper()}"):
                self.process_single_image(image_path, processed_count, total_images)
                processed_count += 1
        
        # Step 5: Create dataset.yaml
        self.create_dataset_yaml()
        
        # Step 6: Print summary
        self.print_summary()
        
        return True
    
    def process_single_image(self, image_path, current_index, total_count):
        """Process a single image: resize and assign to train/val."""
        if current_index % 100 == 0:
            print(f"  Processed {current_index}/{total_count} images...")
        
        try:
            # Check if corresponding label exists
            stem = image_path.stem
            label_path = self.labels_dir / f"{stem}.txt"
            
            if not label_path.exists():
                # Try with different extensions
                for ext in ['.txt', '.TXT']:
                    test_path = self.labels_dir / f"{stem}{ext}"
                    if test_path.exists():
                        label_path = test_path
                        break
            
            if not label_path.exists():
                # No matching label
                return
            
            self.stats['matching_pairs'] += 1
            
            # Determine if this goes to train or val
            if random.random() < self.train_ratio:
                img_dest_dir = self.train_img_dir
                lbl_dest_dir = self.train_lbl_dir
                self.stats['train_count'] += 1
                split_name = "train"
            else:
                img_dest_dir = self.val_img_dir
                lbl_dest_dir = self.val_lbl_dir
                self.stats['val_count'] += 1
                split_name = "val"
            
            # Check original size
            original_size_mb = self.get_file_size_mb(image_path)
            needs_resizing = original_size_mb > self.max_size_mb
            
            # Output paths
            # Keep original extension for compatibility
            output_img_path = img_dest_dir / image_path.name
            output_lbl_path = lbl_dest_dir / f"{stem}.txt"
            
            if needs_resizing:
                # Resize image
                success = self.resize_image(image_path, output_img_path)
                if success:
                    self.stats['resized_images'] += 1
                else:
                    # Fallback: copy original
                    shutil.copy2(image_path, output_img_path)
            else:
                # Copy image as-is
                shutil.copy2(image_path, output_img_path)
                self.stats['copied_images'] += 1
            
            # Copy label
            self.copy_label_file(label_path, output_lbl_path)
            
        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {e}")
            self.stats['errors'] += 1
    
    def create_dataset_yaml(self):
        """Create YOLO dataset.yaml configuration file."""
        yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated by YOLODatasetProcessor
# Images resized to max {self.max_size_mb}MB, target {self.target_width}x{self.target_height}

path: {self.output_dir.absolute()}
train: train/images
val: val/images

# Number of classes
nc: 1

# Class names
names:
  0: eggplant

# Statistics
# Total pairs: {self.stats['matching_pairs']}
# Train: {self.stats['train_count']}
# Val: {self.stats['val_count']}
# Resized: {self.stats['resized_images']}
# Copied: {self.stats['copied_images']}
# Errors: {self.stats['errors']}

# Settings
max_size_mb: {self.max_size_mb}
target_size: [{self.target_width}, {self.target_height}]
train_ratio: {self.train_ratio}
random_seed: {self.seed}
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return yaml_path
    
    def print_summary(self):
        """Print summary of processing."""
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE!")
        print("=" * 70)
        
        print(f"\n📊 Statistics:")
        print(f"  Matching pairs: {self.stats['matching_pairs']}")
        print(f"  Train set: {self.stats['train_count']} images")
        print(f"  Val set: {self.stats['val_count']} images")
        print(f"  Resized images: {self.stats['resized_images']}")
        print(f"  Copied as-is: {self.stats['copied_images']}")
        print(f"  Errors: {self.stats['errors']}")
        
        if self.stats['matching_pairs'] > 0:
            train_percent = (self.stats['train_count'] / self.stats['matching_pairs']) * 100
            val_percent = (self.stats['val_count'] / self.stats['matching_pairs']) * 100
            print(f"  Train/Val split: {train_percent:.1f}% / {val_percent:.1f}%")
        
        print(f"\n📁 Output structure:")
        print(f"  {self.output_dir}/")
        print(f"    ├── train/")
        print(f"    │   ├── images/ ({self.stats['train_count']} files)")
        print(f"    │   └── labels/ ({self.stats['train_count']} files)")
        print(f"    ├── val/")
        print(f"    │   ├── images/ ({self.stats['val_count']} files)")
        print(f"    │   └── labels/ ({self.stats['val_count']} files)")
        print(f"    └── dataset.yaml")
        
        print(f"\n✅ Dataset ready for YOLO training!")
        print(f"\n💡 To train YOLOv8:")
        print(f"   yolo train data={self.output_dir}/dataset.yaml model=yolov8n-seg.pt epochs=100")


def main():
    """Main function."""
    # Configuration
    INPUT_DIR = "/home/saib/ml_project/data/downsample-segment/second_yolov8_training"
    OUTPUT_DIR = "/home/saib/ml_project/data/downsample-segment/second_yolov8_training/processed_dataset"
    
    # Create processor
    processor = YOLODatasetProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_size_mb=1.0,           # Resize images larger than 1MB
        target_size=(640, 640),     # Standard YOLO input size
        train_ratio=0.7,           # 70% train, 30% validation
        seed=42
    )
    
    # Process dataset
    try:
        success = processor.process_dataset()
        if not success:
            print("\n❌ Processing failed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()