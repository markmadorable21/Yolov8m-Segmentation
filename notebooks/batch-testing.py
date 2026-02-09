from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

class BatchEggplantTester:
    def __init__(self, model_path='yolov8m-seg-custom.pt'):
        """Initialize with trained model."""
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
    def test_folder(self, input_folder, output_folder, conf_threshold):
        """
        Test all images in a folder.
        
        Args:
            input_folder: Folder containing test images
            output_folder: Where to save results (default: batch_results_<timestamp>)
            conf_threshold: Confidence threshold for detections
        """
        # Setup paths
        input_path = Path(input_folder)
        
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = f"batch_results_{timestamp}"
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = self._find_images(input_path)
        
        if not image_files:
            print(f"No images found in {input_folder}")
            return
        
        print(f"\nFound {len(image_files)} images to process")
        print(f"Output will be saved to: {output_path}")
        
        # Batch processing
        results = []
        total_detections = 0
        total_inference_time = 0
        
        for idx, img_file in enumerate(image_files):
            print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_file.name}")
            
            # Run inference
            start_time = time.time()
            inference_results = self.model(
                str(img_file),
                conf=conf_threshold,
                save=True,
                save_txt=True,
                save_conf=True,
                project=str(output_path),
                name="predictions"
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Process results
            for result in inference_results:
                num_detections = len(result.boxes) if result.boxes else 0
                total_detections += num_detections
                
                # Save individual stats
                img_result = {
                    'filename': img_file.name,
                    'detections': num_detections,
                    'inference_time': inference_time,
                    'confidences': [],
                    'bboxes': [],
                    'mask_areas': []
                }
                
                # Extract confidence scores
                if result.boxes is not None:
                    for box in result.boxes:
                        img_result['confidences'].append(float(box.conf[0].item()))
                        img_result['bboxes'].append(box.xyxy[0].tolist())
                
                # Extract mask areas
                if result.masks is not None:
                    for mask in result.masks.data:
                        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                        area = np.sum(mask_np > 0)
                        img_result['mask_areas'].append(int(area))
                
                results.append(img_result)
                
                print(f"  Detections: {num_detections}")
                print(f"  Time: {inference_time:.2f}s")
                if num_detections > 0:
                    avg_conf = np.mean(img_result['confidences']) if img_result['confidences'] else 0
                    print(f"  Avg confidence: {avg_conf:.3f}")
        
        # Generate summary
        self._generate_summary(results, output_path, total_detections, total_inference_time)
        
        return results
    
    def _find_images(self, folder_path):
        """Find all image files in folder."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        # Also check subdirectories
        for subdir in folder_path.iterdir():
            if subdir.is_dir():
                for ext in image_extensions:
                    image_files.extend(subdir.glob(f'*{ext}'))
                    image_files.extend(subdir.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def _generate_summary(self, results, output_path, total_detections, total_time):
        """Generate summary report."""
        if not results:
            return
        
        # Calculate statistics
        num_images = len(results)
        avg_detections = total_detections / num_images if num_images > 0 else 0
        avg_time = total_time / num_images if num_images > 0 else 0
        
        # Extract all confidences
        all_confidences = []
        for res in results:
            all_confidences.extend(res['confidences'])
        
        # Create summary dictionary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': num_images,
            'total_detections': total_detections,
            'average_detections_per_image': round(avg_detections, 2),
            'total_inference_time_seconds': round(total_time, 2),
            'average_inference_time_seconds': round(avg_time, 2),
            'images_per_second': round(num_images / total_time, 2) if total_time > 0 else 0,
            'confidence_statistics': {
                'min': round(min(all_confidences), 3) if all_confidences else 0,
                'max': round(max(all_confidences), 3) if all_confidences else 0,
                'mean': round(np.mean(all_confidences), 3) if all_confidences else 0,
                'median': round(np.median(all_confidences), 3) if all_confidences else 0
            },
            'detection_distribution': self._get_detection_distribution(results),
            'images_with_no_detections': len([r for r in results if r['detections'] == 0])
        }
        
        # Save summary as JSON
        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BATCH TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Images processed: {summary['total_images']}")
        print(f"Total eggplants detected: {summary['total_detections']}")
        print(f"Average per image: {summary['average_detections_per_image']:.1f}")
        print(f"Total time: {summary['total_inference_time_seconds']:.1f}s")
        print(f"Average time per image: {summary['average_inference_time_seconds']:.2f}s")
        print(f"Speed: {summary['images_per_second']:.1f} images/second")
        print(f"Images with no detections: {summary['images_with_no_detections']}")
        
        if all_confidences:
            print(f"\nConfidence Statistics:")
            print(f"  Min: {summary['confidence_statistics']['min']:.3f}")
            print(f"  Max: {summary['confidence_statistics']['max']:.3f}")
            print(f"  Mean: {summary['confidence_statistics']['mean']:.3f}")
            print(f"  Median: {summary['confidence_statistics']['median']:.3f}")
        
        print(f"\nDetection distribution:")
        for detections, count in summary['detection_distribution'].items():
            percentage = (count / num_images) * 100
            print(f"  {detections} eggplants: {count} images ({percentage:.1f}%)")
        
        print(f"\nDetailed results saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")
    
    def _get_detection_distribution(self, results):
        """Count how many images have 0, 1, 2, etc. detections."""
        distribution = {}
        for res in results:
            detections = res['detections']
            distribution[detections] = distribution.get(detections, 0) + 1
        
        # Sort by number of detections
        return dict(sorted(distribution.items()))

def main():
    # Configuration
    MODEL_PATH = "/home/saib/ml_project/notebooks/yolo_dataset_split/yolov8m-seg-custom.pt"  # Your trained model
    TEST_FOLDER = "/home/saib/ml_project/data/auto annotation/input"  # Folder containing test images
    OUTPUT_FOLDER = "/home/saib/ml_project/data/auto annotation/output"  # Optional: specify output folder
    CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed
    
    # Create tester
    tester = BatchEggplantTester(MODEL_PATH)
    
    # Run batch test
    results = tester.test_folder(
        input_folder=TEST_FOLDER,
        output_folder=OUTPUT_FOLDER,
        conf_threshold=CONFIDENCE_THRESHOLD
    )

if __name__ == "__main__":
    main()