#!/usr/bin/env python3
"""
Hardcoded AutoDistill Annotation for Eggplant
Simple script with hardcoded paths - just run it!
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import sys

# =============== HARDCODED CONFIGURATION ===============
INPUT_FOLDER = "farm_images"           # Your folder with raw images
OUTPUT_FOLDER = "auto_annotated"       # Where annotations will be saved
CONFIDENCE_THRESHOLD = 0.4            # Lower = more detections, Higher = more accurate
USE_MODEL = "yolov8"                  # "yolov8" or "grounding_dino"
YOLO_MODEL_SIZE = "n"                 # n, s, m, l, x
# =======================================================

def check_and_install_dependencies():
    """Check and install required packages."""
    print("Checking dependencies...")
    
    required_packages = [
        "autodistill",
        "autodistill-yolov8",
        "autodistill-grounding-dino",
        "opencv-python",
        "ultralytics"
    ]
    
    import subprocess
    import importlib
    
    for package in required_packages:
        try:
            # Try to import
            if "autodistill" in package:
                module_name = package.replace("-", "_")
            else:
                module_name = package
            
            importlib.import_module(module_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")

class SimpleEggplantAnnotator:
    def __init__(self):
        """Initialize with hardcoded settings."""
        print("\n" + "="*60)
        print("EGGPLANT AUTO-ANNOTATOR")
        print("="*60)
        
        # Check dependencies first
        check_and_install_dependencies()
        
        # Import after installation
        from autodistill_yolov8 import YOLOv8
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology
        
        print("\nInitializing model...")
        
        # Define what to look for
        self.ontology = CaptionOntology({
            "eggplant": "eggplant",
            "purple eggplant": "eggplant", 
            "aubergine": "eggplant",
            "fresh eggplant": "eggplant"
        })
        
        # Choose model
        if USE_MODEL == "grounding_dino":
            print("Using Grounding DINO model...")
            self.model = GroundingDINO(ontology=self.ontology)
        else:
            print(f"Using YOLOv8-{YOLO_MODEL_SIZE} model...")
            self.model = YOLOv8(ontology=self.ontology, model_size=YOLO_MODEL_SIZE)
        
        print("Model ready!")
    
    def run(self):
        """Run the entire annotation pipeline."""
        print(f"\nInput folder: {INPUT_FOLDER}")
        print(f"Output folder: {OUTPUT_FOLDER}")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        
        # Check input folder
        input_path = Path(INPUT_FOLDER)
        if not input_path.exists():
            print(f"\n❌ ERROR: Input folder '{INPUT_FOLDER}' not found!")
            print(f"Please create a folder named '{INPUT_FOLDER}' and put your images in it.")
            return
        
        # Create output structure
        output_path = Path(OUTPUT_FOLDER)
        output_path.mkdir(exist_ok=True)
        
        folders = {
            "images": output_path / "images",
            "labels": output_path / "labels", 
            "previews": output_path / "previews",
            "failed": output_path / "failed"
        }
        
        for folder in folders.values():
            folder.mkdir(exist_ok=True)
        
        # Get all images
        images = self._find_images(input_path)
        
        if not images:
            print(f"\n❌ No images found in '{INPUT_FOLDER}'!")
            print("Supported formats: .jpg, .jpeg, .png, .bmp")
            return
        
        print(f"\nFound {len(images)} images to process")
        print("Starting annotation...\n")
        
        # Statistics
        stats = {
            "processed": 0,
            "annotated": 0,
            "failed": 0,
            "total_eggplants": 0
        }
        
        # Process each image
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}")
            
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  ⚠ Could not read image")
                    shutil.copy(str(img_path), folders["failed"] / img_path.name)
                    stats["failed"] += 1
                    continue
                
                # Run detection
                results = self.model.predict(
                    str(img_path),
                    confidence=CONFIDENCE_THRESHOLD
                )
                
                # Filter for eggplants only
                eggplants = [r for r in results if r[0] == "eggplant"]
                
                if not eggplants:
                    print(f"  ❌ No eggplants detected")
                    shutil.copy(str(img_path), folders["failed"] / img_path.name)
                    stats["failed"] += 1
                    continue
                
                # Save image
                new_img_name = f"eggplant_{i:04d}.jpg"
                cv2.imwrite(str(folders["images"] / new_img_name), img)
                
                # Create YOLO annotations
                base_name = f"eggplant_{i:04d}"
                self._save_yolo_annotations(img, eggplants, folders["labels"] / f"{base_name}.txt")
                
                # Create preview
                self._create_preview(img, eggplants, folders["previews"] / f"{base_name}_preview.jpg")
                
                # Update stats
                stats["processed"] += 1
                stats["annotated"] += 1
                stats["total_eggplants"] += len(eggplants)
                
                print(f"  ✅ Found {len(eggplants)} eggplants")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}...")
                if 'img_path' in locals():
                    shutil.copy(str(img_path), folders["failed"] / img_path.name)
                stats["failed"] += 1
        
        # Create dataset.yaml
        self._create_yaml_config(output_path)
        
        # Print summary
        self._print_summary(stats, output_path)
        
        print("\n" + "="*60)
        print("ALL DONE! 🎉")
        print("="*60)
    
    def _find_images(self, folder):
        """Find all images in folder."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        images = []
        
        for ext in extensions:
            images.extend(folder.glob(f"*{ext}"))
            images.extend(folder.glob(f"*{ext.upper()}"))
        
        # Also check subfolders
        for item in folder.iterdir():
            if item.is_dir():
                for ext in extensions:
                    images.extend(item.glob(f"*{ext}"))
                    images.extend(item.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def _save_yolo_annotations(self, image, detections, output_path):
        """Save detections in YOLO format."""
        h, w = image.shape[:2]
        
        with open(output_path, 'w') as f:
            for label, (x1, y1, x2, y2), confidence in detections:
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                
                # Class ID 0 for eggplant
                line = f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
                f.write(line)
    
    def _create_preview(self, image, detections, output_path):
        """Create preview image with bounding boxes."""
        preview = image.copy()
        
        # Draw each detection
        for i, (label, (x1, y1, x2, y2), confidence) in enumerate(detections):
            # Different colors for different instances
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
            
            # Draw box
            cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label_text = f"Eggplant {confidence:.2f}"
            cv2.putText(preview, label_text, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add counter
        cv2.putText(preview, f"Eggplants: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imwrite(str(output_path), preview)
    
    def _create_yaml_config(self, output_path):
        """Create YOLO dataset.yaml file."""
        yaml_content = f"""# Auto-generated Eggplant Dataset
path: {output_path.absolute()}
train: images
val: images

# Number of classes
nc: 1

# Class names
names: ['eggplant']

# Created by AutoDistill
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        config_path = output_path / "dataset.yaml"
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created dataset config: {config_path}")
    
    def _print_summary(self, stats, output_path):
        """Print annotation summary."""
        print(f"\n" + "="*60)
        print("ANNOTATION SUMMARY")
        print("="*60)
        
        total = stats["processed"] + stats["failed"]
        
        if total == 0:
            print("No images processed!")
            return
        
        print(f"Total images: {total}")
        print(f"Successfully annotated: {stats['annotated']} ({stats['annotated']/total*100:.1f}%)")
        print(f"Failed (no detections): {stats['failed']} ({stats['failed']/total*100:.1f}%)")
        
        if stats["annotated"] > 0:
            print(f"Total eggplants found: {stats['total_eggplants']}")
            print(f"Average per image: {stats['total_eggplants']/stats['annotated']:.1f}")
        
        print(f"\nOutput saved to: {output_path}/")
        print("Folder structure:")
        print(f"  📁 images/     - All annotated images")
        print(f"  📁 labels/     - YOLO annotation files (.txt)")
        print(f"  📁 previews/   - Preview images with boxes")
        print(f"  📁 failed/     - Images with no detections")
        print(f"  📄 dataset.yaml - YOLO training configuration")
        
        print(f"\nReady for YOLO training! Use this command:")
        print(f"yolo segment train data={output_path}/dataset.yaml model=yolov8n-seg.pt epochs=100")

def main():
    """Main function - just run it!"""
    print("Starting Eggplant Auto-Annotation...")
    
    try:
        annotator = SimpleEggplantAnnotator()
        annotator.run()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have an internet connection (models need to download)")
        print("2. Create a folder named 'farm_images' with your eggplant photos")
        print("3. Run with: python auto_annotate_simple.py")
        print("4. If GPU memory error, reduce image sizes or batch size")

if __name__ == "__main__":
    main()