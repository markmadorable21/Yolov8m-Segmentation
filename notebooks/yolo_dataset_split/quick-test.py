# quick_test_enhanced.py
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def test_model(model_path, image_path):
    """
    Test trained YOLO segmentation model in WSL.
    """ 
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing image: {image_path}")
    
    # Run inference
    results = model(image_path, save=True, save_txt=True, save_conf=True)
    
    # Process results
    for i, result in enumerate(results):
        print(f"\n{'='*50}")
        print(f"IMAGE: {Path(image_path).name}")
        print(f"{'='*50}")
        
        # Detection info
        num_detections = len(result.boxes) if result.boxes else 0
        print(f"Total detections: {num_detections}")
        
        if result.boxes is not None:
            print("\nDetailed detections:")
            for j, box in enumerate(result.boxes):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                bbox = [round(x, 2) for x in box.xyxy[0].tolist()]
                
                print(f"  Eggplant {j+1}:")
                print(f"    Confidence: {conf:.3f}")
                print(f"    Class ID: {cls}")
                print(f"    Bounding Box: {bbox}")
                print(f"    Width: {bbox[2]-bbox[0]:.0f}px, Height: {bbox[3]-bbox[1]:.0f}px")
        
        # Segmentation info
        if result.masks is not None:
            print(f"\nSegmentation masks: {len(result.masks.data)}")
            
            # Create masks directory
            mask_dir = Path('output_masks')
            mask_dir.mkdir(exist_ok=True)
            
            for j, mask in enumerate(result.masks.data):
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                
                # Save mask
                mask_path = mask_dir / f"{Path(image_path).stem}_mask_{j}.png"
                cv2.imwrite(str(mask_path), mask_np)
                
                # Calculate mask area
                mask_area = np.sum(mask_np > 0)
                print(f"  Mask {j}: Area = {mask_area:,} pixels ({mask_path.name})")
        
        # Save annotated image with custom name
        annotated = result.plot()
        output_path = f"annotated_{Path(image_path).name}"
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved as: {output_path}")
    
    print(f"\nAll results saved in: runs/segment/predict/")

if __name__ == "__main__":
    # Test with your model and image
    test_model('yolov8m-seg-custom.pt', '2.jpg')