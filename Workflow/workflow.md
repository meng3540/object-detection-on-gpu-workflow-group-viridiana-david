# Object Detection Workflow

## Overview

This workflow explains the steps used to run real-time object detection on a GPU edge device. The system uses a live camera feed, a YOLOv8 pre-trained model, PyTorch, CUDA, and OpenCV.

## System Block Diagram

```text
USB Camera
   ↓
OpenCV Frame Capture
   ↓
YOLOv8 Pre-trained Model
   ↓
PyTorch + CUDA GPU Inference
   ↓
Annotated Output Frame
   ↓
Display with Bounding Boxes and Class Labels
```

---

## System Setup
The development environment was prepared beforehand on the Scaffold Assignment. Additionally to that setup, new tools were implemented.
First, the system is updated and python3 was installed:

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip libopenblas-dev git
```

Later, Jetson containers were used to avoid compatibility problems between PyTorch, CUDA, and the Jetson software environment.

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
sudo reboot
```

The PyTorch container was launched using the Jetson container command.

```bash
jetson-containers run $(autotag l4t-pytorch)
```

Inside the container, the Ultralytics package was installed to use YOLOv8.

```bash
pip install ultralytics
```

PyTorch was checked to confirm that CUDA was available.

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Then, the YOLOv8n model was imported:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.to("cuda")
```

OpenCV was used to capture frames from the USB camera.

```python
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
```

Each camera frame was passed through the YOLOv8 model. The model returned the detected objects, bounding boxes, class labels, and confidence values.

```python
results = model(frame, conf=0.5, device="cuda")
annotated_frame = results[0].plot()
```

The annotated frame was displayed in a live window.

```python
cv2.imshow("YOLOv8 Live Detection", annotated_frame)
```
---

## Results
The prototype was able to detect objects from the live camera feed. The output displayed bounding boxes, class labels, and confidence values.
The program was run using both CPU and GPU to analyse the output results.

---

## Performance
The prototype was tested in two different cases: CPU-based inference and GPU-based inference. When the model was running on the CPU, the live detection was laggy and the video response was slower. When the model was running on the GPU, the detection was much smoother and the camera feed responded better in real time.

---

## Challenges
The students had two main challenges. First, TensorFlow was the first framework attempted for this workflow, but compatibility and complexity issues made the setup harder. Because of this, the workflow was changed to YOLOv8 with PyTorch.

Additionally, the program could not run the GPU correctly at first, so several tools had to be uninstalled and installed again to make sure the environment was downloaded and configured correctly.

Other challenges included:

- Making sure CUDA was detected by PyTorch.
- Avoiding compatibility issues between PyTorch, Torchvision, and CUDA.
- Installing YOLOv8 correctly inside the container.
- Confirming that the camera feed was being read correctly.

---

## Discussion
This workflow demonstrates how object detection can be treated as a parallel computing problem. YOLOv8 uses many mathematical operations that can be processed faster by the GPU. By using CUDA and PyTorch on the Jetson device, the system can perform inference more efficiently than CPU-only processing.
