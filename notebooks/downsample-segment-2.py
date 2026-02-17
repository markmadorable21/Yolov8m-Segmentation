#!/usr/bin/env python3
"""
Integrated Downsampling and Batch Testing Pipeline
1. Check images for size, downsample if > 1MB
2. Store all processed images in single folder
3. Run YOLO segmentation on downsampled images
4. Output to folder with 3 subfolders:
   - annotated_images (with bounding boxes)
   - yolo_labels (YOLOv8 compatible labels)
   - segmented_eggplants (isolated eggplants, no background)
5. Clean up empty annotation files and their corresponding images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import json 
import time
from datetime import datetime
import glob
from ultralytics import YOLO

class ImageDownsampler:
    def __init__(self, input_folder, output_folder, max_size_mb=1.0, max_dimension=1600, quality=85):
        """
        Initialize image downsampler.
        
        Args:
            input_folder: Folder with images to check
            output_folder: Where to save downsampled images
            max_size_mb: Maximum file size in MB (default: 1.0)
            max_dimension: Maximum width/height in pixels (default: 1600)
            quality: JPEG quality for resized images (1-100, default: 85)
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.max_size_mb = max_size_mb
        self.max_dimension = max_dimension
        self.quality = quality
        
        # Create main output directory
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("IMAGE DOWNSAMPLER")
        print("="*70)
        print(f"Input: {self.input_folder}")
        print(f"Output: {self.output_folder}")
        print(f"Max file size: {max_size_mb} MB")
        print(f"Max dimension: {max_dimension} px")
        print(f"JPEG quality: {quality}")
        print("="*70)
    
    def get_all_images(self):
        """Get all image files from input folder and subfolders."""
        extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', 
                     '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF',
                     '.webp', '.WEBP']
        
        image_files = []
        for ext in extensions:
            pattern = str(self.input_folder / '**' / f'*{ext}')
            files = glob.glob(pattern, recursive=True)
            image_files.extend([Path(f) for f in files])
        
        return sorted(image_files)
    
    def get_file_size_mb(self, file_path):
        """Get file size in megabytes."""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def resize_image(self, img, target_width, target_height):
        """Resize image maintaining aspect ratio."""
        h, w = img.shape[:2]
        
        # Calculate scaling
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Make dimensions divisible by 32 for YOLO
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32
        
        # Ensure minimum size
        new_w = max(new_w, 320)
        new_h = max(new_h, 320)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized, new_w, new_h
    
    def process_single_image(self, img_path, output_path):
        """Process a single image: check size and downsample if needed."""
        filename = img_path.name
        file_ext = img_path.suffix.lower()
        
        print(f"Processing: {filename}")
        
        try:
            # Get original file size
            orig_size_mb = self.get_file_size_mb(img_path)
            print(f"  Original: {orig_size_mb:.2f} MB")
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ❌ Cannot read image")
                return {'status': 'failed', 'reason': 'cannot_read'}
            
            h, w = img.shape[:2]
            print(f"  Dimensions: {w}x{h} pixels")
            
            # Check if already under limits
            if orig_size_mb <= self.max_size_mb and max(w, h) <= self.max_dimension:
                print(f"  ✓ Already under limits - copying")
                shutil.copy2(img_path, output_path)
                return {
                    'status': 'ok',
                    'action': 'copied',
                    'original_mb': orig_size_mb,
                    'final_mb': orig_size_mb,
                    'dimensions': f"{w}x{h}"
                }
            
            # Calculate target dimensions
            target_w, target_h = w, h
            
            # Resize if dimensions exceed maximum
            if w > self.max_dimension or h > self.max_dimension:
                scale = min(self.max_dimension / w, self.max_dimension / h)
                target_w = int(w * scale)
                target_h = int(h * scale)
                print(f"  Resizing to fit dimension limit: {target_w}x{target_h}")
            
            # If still too large, resize further
            if orig_size_mb > self.max_size_mb:
                # Calculate scale needed for file size
                current_pixels = w * h
                target_bytes = self.max_size_mb * 1024 * 1024
                
                # Approximate bytes per pixel
                if self.quality >= 90:
                    bpp = 2.5
                elif self.quality >= 80:
                    bpp = 1.8
                elif self.quality >= 70:
                    bpp = 1.2
                else:
                    bpp = 0.8
                
                target_pixels = target_bytes / bpp
                size_scale = np.sqrt(target_pixels / current_pixels)
                
                # Combine with dimension scale
                final_scale = min(size_scale, 1.0)  # Don't enlarge images
                target_w = int(w * final_scale)
                target_h = int(h * final_scale)
                
                print(f"  Further resizing for file size: {target_w}x{target_h}")
            
            # Ensure minimum dimensions
            target_w = max(target_w, 320)
            target_h = max(target_h, 320)
            
            # Resize image
            if target_w != w or target_h != h:
                img_resized, new_w, new_h = self.resize_image(img, target_w, target_h)
                print(f"  Resized to: {new_w}x{new_h}")
                img_to_save = img_resized
                dimensions_changed = True
            else:
                img_to_save = img
                new_w, new_h = w, h
                dimensions_changed = False
            
            # Save image
            if file_ext in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), img_to_save, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            elif file_ext in ['.png']:
                cv2.imwrite(str(output_path), img_to_save, 
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(str(output_path), img_to_save)
            
            # Check final size
            final_size_mb = self.get_file_size_mb(output_path)
            reduction = ((orig_size_mb - final_size_mb) / orig_size_mb) * 100 if orig_size_mb > 0 else 0
            
            print(f"  Final: {final_size_mb:.2f} MB ({reduction:.1f}% reduction)")
            
            if final_size_mb > self.max_size_mb:
                print(f"  ⚠ Warning: Still exceeds {self.max_size_mb}MB limit")
                return {
                    'status': 'exceeds_limit',
                    'action': 'resized_still_large',
                    'original_mb': orig_size_mb,
                    'final_mb': final_size_mb,
                    'dimensions': f"{new_w}x{new_h}"
                }
            
            return {
                'status': 'resized',
                'action': 'resized',
                'original_mb': orig_size_mb,
                'final_mb': final_size_mb,
                'dimensions': f"{new_w}x{new_h}",
                'reduction_percent': reduction,
                'dimensions_changed': dimensions_changed
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def run_downsampling(self):
        """Run downsampling on all images and return output folder path."""
        image_files = self.get_all_images()
        
        if not image_files:
            print(f"\n❌ No images found in {self.input_folder}")
            return None
        
        print(f"\nFound {len(image_files)} image(s)")
        print("Starting downsampling...\n")
        
        # Statistics
        stats = {
            'total': len(image_files),
            'ok': 0,
            'resized': 0,
            'exceeds_limit': 0,
            'failed': 0,
            'total_original_mb': 0,
            'total_final_mb': 0
        }
        
        results = []
        
        for i, img_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end='')
            
            # Create output path
            output_path = self.output_folder / img_file.name
            
            # Process image
            result = self.process_single_image(img_file, output_path)
            result['file'] = img_file.name
            
            # Update statistics
            stats[result['status']] += 1
            if 'original_mb' in result:
                stats['total_original_mb'] += result['original_mb']
            if 'final_mb' in result:
                stats['total_final_mb'] += result['final_mb']
            
            results.append(result)
        
        # Generate summary
        self.generate_summary(stats, results)
        
        print(f"\n✅ Downsampling complete!")
        print(f"Downsampled images are in: {self.output_folder}")
        
        return self.output_folder
    
    def generate_summary(self, stats, results):
        """Generate and display summary report."""
        print(f"\n{'='*70}")
        print("DOWNSAMPLING SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n📊 Statistics:")
        print(f"  Total images: {stats['total']}")
        print(f"  Already under limit: {stats['ok']}")
        print(f"  Successfully resized: {stats['resized']}")
        print(f"  Still exceeds limit: {stats['exceeds_limit']}")
        print(f"  Failed: {stats['failed']}")
        
        if stats['total'] > 0:
            total_savings = stats['total_original_mb'] - stats['total_final_mb']
            savings_percent = (total_savings / stats['total_original_mb']) * 100 if stats['total_original_mb'] > 0 else 0
            
            print(f"\n💾 Size reduction:")
            print(f"  Original total: {stats['total_original_mb']:.2f} MB")
            print(f"  Final total: {stats['total_final_mb']:.2f} MB")
            print(f"  Total savings: {total_savings:.2f} MB ({savings_percent:.1f}%)")


class EggplantSegmenter:
    def __init__(self, model_path):
        """Initialize YOLO model for segmentation."""
        print(f"\nLoading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    
    def cleanup_empty_annotations(self, labels_dir, images_dir):
        """
        Clean up empty annotation files and their corresponding images.
        
        Args:
            labels_dir: Directory containing YOLO label files (.txt)
            images_dir: Directory containing corresponding images
            
        Returns:
            tuple: (deleted_labels_count, deleted_images_count)
        """
        print(f"\n{'='*70}")
        print("CLEANING UP EMPTY ANNOTATIONS")
        print(f"{'='*70}")
        
        labels_path = Path(labels_dir)
        images_path = Path(images_dir)
        
        # Get all text files in labels directory
        txt_files = list(labels_path.glob("*.txt"))
        
        if not txt_files:
            print("No label files found to check.")
            return 0, 0
        
        print(f"Found {len(txt_files)} label files to check.")
        
        deleted_labels = 0
        deleted_images = 0
        
        for txt_file in txt_files:
            try:
                # Check if file is empty (0 bytes)
                if os.path.getsize(txt_file) == 0:
                    # Get corresponding image filename (same name, different extension)
                    base_name = txt_file.stem
                    
                    # Look for corresponding image in images directory
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
                                       '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF', '.WEBP']
                    
                    matching_images = []
                    for ext in image_extensions:
                        potential_image = images_path / f"{base_name}{ext}"
                        if potential_image.exists():
                            matching_images.append(potential_image)
                    
                    # Delete the empty label file
                    txt_file.unlink()
                    deleted_labels += 1
                    print(f"  Deleted empty label: {txt_file.name}")
                    
                    # Delete all matching images
                    for img_file in matching_images:
                        img_file.unlink()
                        deleted_images += 1
                        print(f"  Deleted corresponding image: {img_file.name}")
                        
                else:
                    # Check if file contains only whitespace or empty lines
                    with open(txt_file, 'r') as f:
                        content = f.read().strip()
                    
                    if not content:
                        # File is essentially empty (only whitespace)
                        base_name = txt_file.stem
                        
                        # Look for corresponding images
                        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
                                           '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF', '.WEBP']
                        
                        matching_images = []
                        for ext in image_extensions:
                            potential_image = images_path / f"{base_name}{ext}"
                            if potential_image.exists():
                                matching_images.append(potential_image)
                        
                        # Delete the essentially empty label file
                        txt_file.unlink()
                        deleted_labels += 1
                        print(f"  Deleted whitespace-only label: {txt_file.name}")
                        
                        # Delete all matching images
                        for img_file in matching_images:
                            img_file.unlink()
                            deleted_images += 1
                            print(f"  Deleted corresponding image: {img_file.name}")
                            
            except Exception as e:
                print(f"  Error processing {txt_file.name}: {e}")
                continue
        
        print(f"\nCleanup summary:")
        print(f"  Deleted {deleted_labels} empty label files")
        print(f"  Deleted {deleted_images} corresponding images")
        
        return deleted_labels, deleted_images
    
    def create_mask_from_data(self, mask_data, image_shape):
        """
        Create binary mask from YOLO mask data.
        
        Args:
            mask_data: Mask tensor from YOLO
            image_shape: Original image shape
            
        Returns:
            Binary mask
        """
        h, w = image_shape[:2]
        
        # Convert mask tensor to numpy array
        if mask_data is not None:
            try:
                mask_np = mask_data.cpu().numpy()
                # Resize mask to original image dimensions
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                # Convert to binary mask
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                return binary_mask
            except Exception as e:
                print(f"    ⚠ Error processing mask data: {e}")
        
        return np.zeros((h, w), dtype=np.uint8)
    
    def extract_isolated_eggplant(self, image, mask, bbox):
        """
        Extract isolated eggplant with transparent background.
        
        Args:
            image: Original image
            mask: Binary mask of eggplant (full image size)
            bbox: Bounding box [x1, y1, x2, y2] in pixels
            
        Returns:
            Image with isolated eggplant on transparent background
        """
        # Ensure bbox coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Clip coordinates to image boundaries
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Check if bbox is valid
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop the region
        cropped = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        
        # Check if cropped region is valid
        if cropped.size == 0 or cropped_mask.size == 0:
            return None
        
        # Create BGRA image (with alpha channel)
        if len(cropped.shape) == 3:
            h_crop, w_crop, c = cropped.shape
            result = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
            result[:, :, :3] = cropped
            result[:, :, 3] = cropped_mask
            # Make background transparent where mask is 0
            result[cropped_mask == 0] = [0, 0, 0, 0]
        else:
            # Grayscale image
            h_crop, w_crop = cropped.shape
            result = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
            result[:, :, :3] = np.stack([cropped]*3, axis=-1)
            result[:, :, 3] = cropped_mask
            result[cropped_mask == 0] = [0, 0, 0, 0]
        
        return result
    
    def process_images(self, input_folder, output_folder, conf_threshold=0.5, cleanup=True):
        """
        Process all images in folder for segmentation.
        
        Args:
            input_folder: Folder with downsampled images
            output_folder: Where to save results
            conf_threshold: Confidence threshold
            cleanup: Whether to clean up empty annotations (default: True)
            
        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output subfolders
        annotated_dir = output_path / "annotated_images"
        labels_dir = output_path / "yolo_labels"
        segmented_dir = output_path / "segmented_eggplants"
        
        # Clean existing output
        for dir_path in [output_path, annotated_dir, labels_dir, segmented_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        # Create fresh directories
        annotated_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        segmented_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("EGGPLANT SEGMENTATION")
        print(f"{'='*70}")
        print(f"Input: {input_folder}")
        print(f"Output: {output_folder}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Cleanup empty annotations: {cleanup}")
        print(f"\nOutput structure:")
        print(f"  {annotated_dir}/")
        print(f"  {labels_dir}/")
        print(f"  {segmented_dir}/")
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"\n❌ No images found in {input_folder}")
            return None
        
        print(f"\nFound {len(image_files)} images to process")
        
        # Processing statistics
        results = []
        total_detections = 0
        total_segmented = 0
        total_time = 0
        
        for idx, img_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {img_file.name}")
            
            start_time = time.time()
            
            try:
                # Read image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"  ✗ Cannot read image")
                    continue
                
                h, w = img.shape[:2]
                
                # Run YOLO inference WITH SEGMENTATION ENABLED
                yolo_results = self.model.predict(
                    source=str(img_file),
                    conf=conf_threshold,
                    save=False,
                    save_txt=False,
                    save_conf=False,
                    boxes=True,
                    show=False,
                    verbose=False  # Turn off YOLO's own output
                )
                
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Process each detection
                eggplants_segmented = 0
                
                for result in yolo_results:
                    num_detections = len(result.boxes) if result.boxes else 0
                    total_detections += num_detections
                    
                    # Check if masks exist
                    has_masks = result.masks is not None
                    print(f"  Detections: {num_detections}, Has masks: {has_masks}")
                    
                    # Save annotated image
                    annotated_img = result.plot()
                    annotated_path = annotated_dir / img_file.name
                    cv2.imwrite(str(annotated_path), annotated_img)
                    
                    # Save YOLO labels
                    if result.boxes is not None:
                        self.save_yolo_labels(result, labels_dir, img_file.stem, w, h)
                    
                    # Extract and save segmented eggplants
                    if has_masks:
                        eggplants_segmented = self.save_segmented_eggplants(
                            result, img, segmented_dir, img_file.stem
                        )
                        total_segmented += eggplants_segmented
                    else:
                        print(f"  ⚠ No segmentation masks available for {img_file.name}")
                        print(f"  Model might not support segmentation or masks are empty")
                    
                    # Collect stats
                    img_result = {
                        'filename': img_file.name,
                        'detections': num_detections,
                        'eggplants_segmented': eggplants_segmented,
                        'has_masks': has_masks,
                        'inference_time': inference_time,
                        'confidences': [],
                        'image_path': str(annotated_path)
                    }
                    
                    if result.boxes is not None:
                        for box in result.boxes:
                            img_result['confidences'].append(float(box.conf[0].item()))
                    
                    results.append(img_result)
                    
                    print(f"  Time: {inference_time:.2f}s")
                    
                    if num_detections > 0 and has_masks:
                        print(f"  Segmented eggplants: {eggplants_segmented}")
                        if eggplants_segmented == 0:
                            print(f"  ⚠ Detections found but no eggplants were segmented")
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up empty annotations if requested
        if cleanup:
            deleted_labels, deleted_images = self.cleanup_empty_annotations(labels_dir, annotated_dir)
            
            # Update results to remove deleted files
            if deleted_labels > 0:
                # Filter out results for deleted files
                remaining_results = []
                for result in results:
                    img_path = Path(result['image_path'])
                    if img_path.exists():
                        remaining_results.append(result)
                results = remaining_results
        
        # Generate summary
        self.generate_segmentation_summary(results, output_path, total_detections, 
                                         total_segmented, total_time, cleanup)
        
        return results

    def save_yolo_labels(self, result, labels_dir, base_name, img_w, img_h):
        """Save YOLO format labels as segmentation polygons."""
        label_path = labels_dir / f"{base_name}.txt"
        
        with open(label_path, 'w') as f:
            if result.masks is not None and result.boxes is not None:
                # Get masks data - masks.xy is already a list of numpy arrays
                masks_xy = result.masks.xy
                
                for i, box in enumerate(result.boxes):
                    try:
                        # Get class ID (need to convert tensor to numpy)
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get corresponding mask polygon
                        if i < len(masks_xy):
                            mask_polygon = masks_xy[i]  # Already a numpy array
                            
                            # Flatten polygon to 1D array: [x1, y1, x2, y2, ...]
                            polygon_flat = mask_polygon.flatten()
                            
                            # Normalize coordinates (masks.xy are in pixel coordinates)
                            polygon_normalized = polygon_flat / np.array([img_w, img_h] * (len(polygon_flat) // 2))
                            
                            # Format: class_id x1 y1 x2 y2 ...
                            polygon_str = ' '.join([f'{coord:.17f}' for coord in polygon_normalized])
                            f.write(f"{cls} {polygon_str}\n")
                        else:
                            # Fallback to bounding box if no mask
                            xywh = box.xywhn[0].cpu().numpy()
                            f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")
                            
                    except Exception as e:
                        print(f"Warning: Error saving polygon label: {e}")
                        # Fallback to bounding box
                        try:
                            xywh = box.xywhn[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")
                        except:
                            # Last resort: use pixel coordinates
                            xyxy = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            x1, y1, x2, y2 = xyxy
                            x_center = (x1 + x2) / 2 / img_w
                            y_center = (y1 + y2) / 2 / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            elif result.boxes is not None:
                # If no masks, save bounding boxes
                for box in result.boxes:
                    try:
                        xywh = box.xywhn[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")
                    except:
                        xyxy = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        x1, y1, x2, y2 = xyxy
                        x_center = (x1 + x2) / 2 / img_w
                        y_center = (y1 + y2) / 2 / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def save_segmented_eggplants(self, result, original_img, segmented_dir, base_name):
        """Save isolated eggplants with transparent background."""
        eggplants_saved = 0
        
        if result.masks is not None and result.boxes is not None:
            # Get masks and boxes
            masks = result.masks.data
            boxes = result.boxes
            
            # Get image dimensions
            h, w = original_img.shape[:2]
            
            # Process each detection
            for i in range(len(boxes)):
                try:
                    # Get mask data
                    mask_data = masks[i] if i < len(masks) else None
                    box = boxes[i]
                    
                    # Get bounding box
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Create binary mask from mask data
                    mask = self.create_mask_from_data(mask_data, original_img.shape)
                    
                    # Check if mask is not empty
                    if np.sum(mask) == 0:
                        print(f"    ⚠ Empty mask for detection {i+1}")
                        continue
                    
                    # Extract isolated eggplant
                    isolated_eggplant = self.extract_isolated_eggplant(original_img, 
                                                                     mask, 
                                                                     bbox)
                    
                    if isolated_eggplant is None:
                        print(f"    ⚠ Failed to extract eggplant {i+1}")
                        continue
                    
                    # Check if extracted eggplant has non-zero alpha pixels
                    alpha_channel = isolated_eggplant[:, :, 3]
                    non_zero_pixels = np.sum(alpha_channel > 0)
                    
                    if non_zero_pixels == 0:
                        print(f"    ⚠ Extracted eggplant {i+1} has only transparent pixels")
                        continue
                    
                    # Save as PNG with transparency
                    output_path = segmented_dir / f"{base_name}_eggplant_{i+1}.png"
                    success = cv2.imwrite(str(output_path), isolated_eggplant)
                    
                    if success:
                        eggplants_saved += 1
                        print(f"    ✓ Saved isolated eggplant {i+1} ({non_zero_pixels} non-transparent pixels)")
                    else:
                        print(f"    ⚠ Failed to save image {output_path}")
                    
                except Exception as e:
                    print(f"    ⚠ Error segmenting eggplant {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        return eggplants_saved
    
    def generate_segmentation_summary(self, results, output_path, total_detections, 
                                    total_segmented, total_time, cleanup_done=True):
        """Generate segmentation summary report."""
        if not results:
            print("No results to summarize!")
            return
        
        num_images = len(results)
        avg_detections = total_detections / num_images if num_images > 0 else 0
        avg_segmented = total_segmented / num_images if num_images > 0 else 0
        avg_time = total_time / num_images if num_images > 0 else 0
        
        # Collect statistics about masks
        images_with_masks = sum(1 for r in results if r.get('has_masks', False))
        images_with_detections_no_masks = sum(1 for r in results if r['detections'] > 0 and not r.get('has_masks', False))
        
        # Collect all confidences
        all_confidences = []
        for res in results:
            all_confidences.extend(res['confidences'])
        
        # Count actual files after cleanup
        annotated_dir = output_path / "annotated_images"
        labels_dir = output_path / "yolo_labels"
        segmented_dir = output_path / "segmented_eggplants"
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': num_images,
            'total_detections': total_detections,
            'total_segmented_eggplants': total_segmented,
            'images_with_masks': images_with_masks,
            'images_with_detections_but_no_masks': images_with_detections_no_masks,
            'average_detections_per_image': round(avg_detections, 2),
            'average_segmented_per_image': round(avg_segmented, 2),
            'total_processing_time_seconds': round(total_time, 2),
            'average_time_per_image': round(avg_time, 2),
            'images_per_second': round(num_images / total_time, 2) if total_time > 0 else 0,
            'cleanup_performed': cleanup_done,
            'confidence_statistics': {
                'min': round(min(all_confidences), 3) if all_confidences else 0,
                'max': round(max(all_confidences), 3) if all_confidences else 0,
                'mean': round(np.mean(all_confidences), 3) if all_confidences else 0,
                'median': round(np.median(all_confidences), 3) if all_confidences else 0
            },
            'images_with_no_detections': len([r for r in results if r['detections'] == 0]),
            'output_files': {
                'annotated_images': len(list(annotated_dir.glob("*"))) if annotated_dir.exists() else 0,
                'yolo_labels': len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0,
                'segmented_eggplants': len(list(segmented_dir.glob("*.png"))) if segmented_dir.exists() else 0
            }
        }
        
        # Save summary as JSON
        summary_path = output_path / "segmentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print("SEGMENTATION SUMMARY")
        print(f"{'='*70}")
        print(f"Images processed: {summary['total_images']}")
        print(f"Total detections: {summary['total_detections']}")
        print(f"Total segmented eggplants: {summary['total_segmented_eggplants']}")
        print(f"Images with masks available: {summary['images_with_masks']}")
        print(f"Images with detections but no masks: {summary['images_with_detections_but_no_masks']}")
        print(f"Average per image: {summary['average_detections_per_image']:.1f} detections, "
              f"{summary['average_segmented_per_image']:.1f} segmented")
        print(f"Total time: {summary['total_processing_time_seconds']:.1f}s")
        print(f"Average time: {summary['average_time_per_image']:.2f}s")
        print(f"Speed: {summary['images_per_second']:.1f} images/second")
        print(f"Images with no detections: {summary['images_with_no_detections']}")
        print(f"Cleanup performed: {summary['cleanup_performed']}")
        
        print(f"\n📁 Output files:")
        print(f"  annotated_images/: {summary['output_files']['annotated_images']} images")
        print(f"  yolo_labels/: {summary['output_files']['yolo_labels']} label files")
        print(f"  segmented_eggplants/: {summary['output_files']['segmented_eggplants']} isolated eggplants")
        
        if all_confidences:
            print(f"\nConfidence Statistics:")
            print(f"  Min: {summary['confidence_statistics']['min']:.3f}")
            print(f"  Max: {summary['confidence_statistics']['max']:.3f}")
            print(f"  Mean: {summary['confidence_statistics']['mean']:.3f}")
            print(f"  Median: {summary['confidence_statistics']['median']:.3f}")
        
        print(f"\n✅ Segmentation complete!")
        print(f"Results saved to: {output_path}")


class IntegratedPipeline:
    def __init__(self):
        """Initialize integrated pipeline."""
        print("="*70)
        print("INTEGRATED EGGPLANT PROCESSING PIPELINE")
        print("="*70)
    
    def run_pipeline(self, config):
        """
        Run complete pipeline: downsampling + segmentation.
        
        Args:
            config: Dictionary with configuration parameters
        """
        print("\n🚀 Starting pipeline...")
        
        # Step 1: Downsampling
        print(f"\n{'='*70}")
        print("STEP 1: IMAGE DOWNSAMPLING")
        print(f"{'='*70}")
        
        downsampler = ImageDownsampler(
            input_folder=config['input_folder'],
            output_folder=config['downsampled_folder'],
            max_size_mb=config['max_size_mb'],
            max_dimension=config['max_dimension'],
            quality=config['quality']
        )
        
        downsampled_folder = downsampler.run_downsampling()
        
        if downsampled_folder is None:
            print("❌ Downsampling failed. Exiting.")
            return
        
        # Step 2: Segmentation
        print(f"\n{'='*70}")
        print("STEP 2: EGGPLANT SEGMENTATION")
        print(f"{'='*70}")
        
        segmenter = EggplantSegmenter(config['model_path'])
        
        results = segmenter.process_images(
            input_folder=downsampled_folder,
            output_folder=config['final_output_folder'],
            conf_threshold=config['confidence_threshold'],
            cleanup=config.get('cleanup_empty_annotations', True)  # Default to True
        )
        
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"📁 Downsampled images: {downsampled_folder}")
        print(f"📁 Final results: {config['final_output_folder']}")
        print(f"  ├── annotated_images/")
        print(f"  ├── yolo_labels/")
        print(f"  └── segmented_eggplants/")
        print(f"\n✅ Pipeline executed successfully!")


def main():
    """Main function with configuration."""
    # ============= CONFIGURATION =============
    CONFIG = {
        'input_folder': "/home/saib/ml_project/data/batch_testing/input",
        'downsampled_folder': "/home/saib/ml_project/data/batch_testing/downsample",
        'final_output_folder': "/home/saib/ml_project/data/batch_testing/output",
        'model_path': "/home/saib/ml_project/data/batch_testing/best.pt",
        'max_size_mb': 1.0,
        'max_dimension': 1600,
        'quality': 90,
        'confidence_threshold': 0.5,
        'cleanup_empty_annotations': True  # Add this option
    }
    # ========================================
    
    # Create pipeline and run
    pipeline = IntegratedPipeline()
    pipeline.run_pipeline(CONFIG)


if __name__ == "__main__":
    main()