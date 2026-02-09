#!/usr/bin/env python3
"""
Check and resize images to ensure none exceed 1MB
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

class ImageSizeOptimizer:
    def __init__(self, input_folder, output_folder=None, max_size_mb=1.0, max_dimension=1600, quality=85):
        """
        Initialize image optimizer.
        
        Args:
            input_folder: Folder with images to check
            output_folder: Where to save optimized images (creates subfolders)
            max_size_mb: Maximum file size in MB (default: 1.0MB)
            max_dimension: Maximum width or height in pixels (default: 1600)
            quality: JPEG quality for resized images (1-100, default: 85)
        """
        self.input_folder = Path(input_folder)
        self.max_size_mb = max_size_mb
        self.max_dimension = max_dimension
        self.quality = quality
        
        # Setup output folder structure
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = self.input_folder.parent / f"{self.input_folder.name}_optimized_{timestamp}"
        else:
            self.output_folder = Path(output_folder)
        
        # Create output subdirectories
        self.dirs = {
            'ok': self.output_folder / 'ok',           # Already under 1MB
            'resized': self.output_folder / 'resized', # Successfully resized
            'failed': self.output_folder / 'failed',   # Failed to process
            'skipped': self.output_folder / 'skipped', # Non-image files
            'exceeds_limit': self.output_folder / 'exceeds_limit' # Still >1MB after resize
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("IMAGE SIZE OPTIMIZER")
        print("="*70)
        print(f"Input: {self.input_folder}")
        print(f"Output: {self.output_folder}")
        print(f"Max file size: {max_size_mb} MB")
        print(f"Max dimension: {max_dimension} px")
        print(f"JPEG quality: {quality}")
        print("="*70)
    
    def get_all_images(self):
        """Get all image files from input folder."""
        extensions = [
            '.jpg', '.jpeg', '.JPG', '.JPEG',
            '.png', '.PNG',
            '.bmp', '.BMP',
            '.tif', '.tiff', '.TIF', '.TIFF',
            '.webp', '.WEBP'
        ]
        
        image_files = []
        for ext in extensions:
            image_files.extend(self.input_folder.rglob(f'*{ext}'))
        
        return sorted(image_files)
    
    def get_file_size_mb(self, file_path):
        """Get file size in megabytes."""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def calculate_target_size(self, width, height, target_mb=1.0):
        """
        Calculate new dimensions to achieve target file size.
        Based on average JPEG compression ratios.
        """
        # Approximate bytes per pixel for JPEG at given quality
        # Lower quality = fewer bytes per pixel
        if self.quality >= 90:
            bytes_per_pixel = 2.5
        elif self.quality >= 80:
            bytes_per_pixel = 1.8
        elif self.quality >= 70:
            bytes_per_pixel = 1.2
        else:
            bytes_per_pixel = 0.8
        
        # Current pixel count
        current_pixels = width * height
        
        # Target bytes
        target_bytes = target_mb * 1024 * 1024
        
        # Calculate target pixels
        target_pixels = target_bytes / bytes_per_pixel
        
        if target_pixels >= current_pixels:
            # Already under target, no need to resize
            return width, height
        
        # Calculate scaling factor
        scale = np.sqrt(target_pixels / current_pixels)
        
        # New dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Ensure minimum dimensions
        new_width = max(new_width, 100)
        new_height = max(new_height, 100)
        
        return new_width, new_height
    
    def resize_image(self, img, width, height):
        """Resize image maintaining aspect ratio."""
        h, w = img.shape[:2]
        
        # Calculate scaling
        scale_w = width / w
        scale_h = height / h
        scale = min(scale_w, scale_h)
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized, new_w, new_h
    
    def optimize_single_image(self, img_path):
        """
        Process a single image:
        1. Check file size
        2. Resize if > max_size_mb
        3. Save with appropriate quality
        """
        filename = img_path.name
        file_ext = img_path.suffix.lower()
        
        print(f"\nProcessing: {filename}")
        
        try:
            # Get original file size
            orig_size_mb = self.get_file_size_mb(img_path)
            print(f"  Original size: {orig_size_mb:.2f} MB")
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ❌ Cannot read image")
                shutil.copy2(img_path, self.dirs['failed'] / filename)
                return {'status': 'failed', 'reason': 'cannot_read', 'original_mb': orig_size_mb}
            
            h, w = img.shape[:2]
            print(f"  Dimensions: {w}x{h} pixels")
            
            # Check if already under limit
            if orig_size_mb <= self.max_size_mb and max(w, h) <= self.max_dimension:
                print(f"  ✓ Already under limits")
                shutil.copy2(img_path, self.dirs['ok'] / filename)
                return {
                    'status': 'ok',
                    'original_mb': orig_size_mb,
                    'final_mb': orig_size_mb,
                    'dimensions': f"{w}x{h}",
                    'action': 'copied'
                }
            
            # Determine target dimensions
            if orig_size_mb > self.max_size_mb:
                # Calculate dimensions to achieve target file size
                target_w, target_h = self.calculate_target_size(w, h, self.max_size_mb)
                print(f"  Target for {self.max_size_mb}MB: {target_w}x{target_h}")
            else:
                # Just limit dimensions
                target_w, target_h = self.max_dimension, self.max_dimension
            
            # Apply dimension limits
            target_w = min(target_w, self.max_dimension)
            target_h = min(target_h, self.max_dimension)
            
            # Resize if needed
            if w > target_w or h > target_h:
                img_resized, new_w, new_h = self.resize_image(img, target_w, target_h)
                print(f"  Resized to: {new_w}x{new_h}")
                img_to_save = img_resized
            else:
                img_to_save = img
                new_w, new_h = w, h
            
            # Save with appropriate settings
            output_path = self.dirs['resized'] / filename
            
            if file_ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                # JPEG - use quality parameter
                cv2.imwrite(str(output_path), img_to_save, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            elif file_ext in ['.png', '.PNG']:
                # PNG - compression level 3 (balanced)
                cv2.imwrite(str(output_path), img_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                # Other formats - save as is
                cv2.imwrite(str(output_path), img_to_save)
            
            # Check final file size
            final_size_mb = self.get_file_size_mb(output_path)
            reduction = ((orig_size_mb - final_size_mb) / orig_size_mb) * 100
            
            print(f"  Final size: {final_size_mb:.2f} MB ({reduction:.1f}% reduction)")
            
            if final_size_mb > self.max_size_mb:
                print(f"  ⚠ Still exceeds {self.max_size_mb}MB limit!")
                exceeds_path = self.dirs['exceeds_limit'] / filename
                shutil.move(output_path, exceeds_path)
                
                return {
                    'status': 'exceeds_limit',
                    'original_mb': orig_size_mb,
                    'final_mb': final_size_mb,
                    'dimensions': f"{new_w}x{new_h}",
                    'action': 'resized_still_large'
                }
            
            return {
                'status': 'resized',
                'original_mb': orig_size_mb,
                'final_mb': final_size_mb,
                'dimensions': f"{new_w}x{new_h}",
                'action': 'resized',
                'reduction_percent': reduction
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            try:
                shutil.copy2(img_path, self.dirs['failed'] / filename)
            except:
                pass
            return {'status': 'failed', 'reason': str(e), 'original_mb': orig_size_mb}
    
    def run_optimization(self):
        """Run optimization on all images."""
        image_files = self.get_all_images()
        
        if not image_files:
            print(f"\n❌ No images found in {self.input_folder}")
            return None
        
        print(f"\nFound {len(image_files)} image(s)")
        print("Starting optimization...\n")
        
        # Statistics
        stats = {
            'total': len(image_files),
            'ok': 0,
            'resized': 0,
            'exceeds_limit': 0,
            'failed': 0,
            'skipped': 0,
            'total_original_mb': 0,
            'total_final_mb': 0
        }
        
        results = []
        
        for i, img_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end='')
            result = self.optimize_single_image(img_file)
            
            # Update statistics
            stats[result['status']] += 1
            if 'original_mb' in result:
                stats['total_original_mb'] += result['original_mb']
            if 'final_mb' in result:
                stats['total_final_mb'] += result['final_mb']
            
            results.append({
                'file': img_file.name,
                **result
            })
        
        # Generate summary
        self.generate_summary(stats, results)
        
        return results
    
    def generate_summary(self, stats, results):
        """Generate and display summary report."""
        print(f"\n{'='*70}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n📊 Statistics:")
        print(f"  Total images: {stats['total']}")
        print(f"  Already under {self.max_size_mb}MB: {stats['ok']}")
        print(f"  Successfully resized: {stats['resized']}")
        print(f"  Still exceeds limit: {stats['exceeds_limit']}")
        print(f"  Failed to process: {stats['failed']}")
        
        if stats['total'] > 0:
            total_savings_mb = stats['total_original_mb'] - stats['total_final_mb']
            total_savings_percent = (total_savings_mb / stats['total_original_mb']) * 100 if stats['total_original_mb'] > 0 else 0
            
            print(f"\n💾 Size reduction:")
            print(f"  Original total: {stats['total_original_mb']:.2f} MB")
            print(f"  Final total: {stats['total_final_mb']:.2f} MB")
            print(f"  Total savings: {total_savings_mb:.2f} MB ({total_savings_percent:.1f}%)")
            print(f"  Average per image: {stats['total_original_mb']/stats['total']:.2f} MB → {stats['total_final_mb']/stats['total']:.2f} MB")
        
        print(f"\n📁 Output structure:")
        for category, dir_path in self.dirs.items():
            num_files = len(list(dir_path.glob('*')))
            if num_files > 0:
                size_mb = sum(os.path.getsize(f) for f in dir_path.glob('*')) / (1024 * 1024)
                print(f"  {dir_path.name}/ - {num_files} files ({size_mb:.1f} MB)")
        
        # List files that still exceed limit
        if stats['exceeds_limit'] > 0:
            print(f"\n⚠ Files still exceeding {self.max_size_mb}MB (need manual attention):")
            exceed_files = [r for r in results if r['status'] == 'exceeds_limit']
            for r in exceed_files[:10]:  # Show first 10
                print(f"  {r['file']}: {r['final_mb']:.2f} MB")
            if len(exceed_files) > 10:
                print(f"  ... and {len(exceed_files) - 10} more")
        
        # Save detailed report
        report_path = self.output_folder / "optimization_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Image Optimization Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input folder: {self.input_folder}\n")
            f.write(f"Max file size: {self.max_size_mb} MB\n")
            f.write(f"Max dimension: {self.max_dimension} px\n")
            f.write(f"JPEG quality: {self.quality}\n\n")
            
            f.write("SUMMARY\n")
            f.write(f"Total images: {stats['total']}\n")
            f.write(f"Already under limit: {stats['ok']}\n")
            f.write(f"Successfully resized: {stats['resized']}\n")
            f.write(f"Still exceeds limit: {stats['exceeds_limit']}\n")
            f.write(f"Failed: {stats['failed']}\n\n")
            
            f.write("DETAILED RESULTS\n")
            for r in results:
                f.write(f"{r['file']}: {r.get('original_mb', 0):.2f}MB → {r.get('final_mb', 0):.2f}MB, Status: {r['status']}\n")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        print(f"\n✅ Optimization complete!")
        print(f"Optimized images ready in: {self.output_folder}")
        print(f"{'='*70}")

def main():
    """Main function with your specific paths."""
    # ============= CONFIGURATION =============
    INPUT_FOLDER = "/home/saib/ml_project/data/manual annotation/input"
    OUTPUT_FOLDER = None  # Auto-generated with timestamp
    MAX_SIZE_MB = 1.0     # Maximum 1MB per file
    MAX_DIMENSION = 1600  # Maximum 1600px width/height
    QUALITY = 85          # JPEG quality (1-100)
    # ========================================
    
    print("Starting image size optimization...")
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return
    
    # Initialize optimizer
    optimizer = ImageSizeOptimizer(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        max_size_mb=MAX_SIZE_MB,
        max_dimension=MAX_DIMENSION,
        quality=QUALITY
    )
    
    # Run optimization
    results = optimizer.run_optimization()
    
    if results:
        print(f"\n🎯 Next steps:")
        print(f"1. Use the 'ok/' folder for images already under {MAX_SIZE_MB}MB")
        print(f"2. Use the 'resized/' folder for optimized images")
        print(f"3. Check 'exceeds_limit/' for files that need manual attention")
        print(f"4. Run your YOLO batch testing on the optimized images")

if __name__ == "__main__":
    main()