# test_video.py
from ultralytics import YOLO
import cv2

def test_video(model_path='yolov8m-seg-custom.pt', video_path='test.mp4'):
    """
    Quick test on video file.
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing video: {video_path}")
    
    # Run inference on video
    results = model.track(
        source=video_path,
        save=True,           # Save output video
        show=False,          # Set to False for WSL
        conf=0.5,            # Confidence threshold
        iou=0.5,             # IoU threshold
        tracker="bytetrack.yaml",  # Object tracking
        stream=True          # Stream mode for video
    )
    
    print("\nProcessing video frames...")
    frame_count = 0
    
    for result in results:
        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames
            detections = len(result.boxes) if result.boxes else 0
            print(f"Frame {frame_count}: {detections} detections")
    
    print(f"\nVideo processing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Output saved to: runs/segment/predict/")

if __name__ == "__main__":
    # Configure paths
    MODEL_PATH = "/home/saib/ml_project/notebooks/yolo_dataset_split/yolov8m-seg-custom.pt"  # Your trained model
    VIDEO_PATH = "/home/saib/ml_project/data/2.mp4"          # Your video file
    
    test_video(MODEL_PATH, VIDEO_PATH)