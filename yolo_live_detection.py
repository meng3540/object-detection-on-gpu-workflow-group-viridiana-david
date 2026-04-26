#!/usr/bin/env python3
"""
Real-time object detection with YOLOv8 on Jetson
Requires: ultralytics, opencv-python, torch
"""
import torch
import cv2
import torch
from ultralytics import YOLO

# Configuration
MODEL_PATH = 'yolov8n.pt'
CONF_THRESHOLD = 0.5
CAMERA_ID = 0
INPUT_SIZE = 640

def main():
    # Check GPU
    print(f"GPU Active: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model = YOLO(MODEL_PATH).to(device)
    if device == 'cuda':
        model.half() # FP16 optimization
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_SIZE)
    
    print("Press 'q' to quit | 's' to save screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference
        results = model(frame, conf=CONF_THRESHOLD, device=device)
        
        # Render detections
        annotated = results[0].plot()
        
        # Display
        cv2.imshow('YOLOv8 Detection', annotated)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('detection.jpg', annotated)
            print("Screenshot saved")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
