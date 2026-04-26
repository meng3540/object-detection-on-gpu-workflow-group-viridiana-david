# Parallel-Programming--Object-Detection-inferencing-on-GPU-edge-devices
# Object Detection Workflow on GPU Edge Device
### YOLOv8 Live Object Detection on Jetson Edge Device
Team: Edgar David Medellin Flores &amp; Viridiana Espiritu Sanchez 

##  Project Overview
This project aims to develop and build a workflow for real-time object identification on a GPU-accelerated embedded system. The goal is to show that AI inference can be performed efficiently on edge devices utilizing parallel computing approaches.

---

## Problem Statement
Object detection involves detecting and locating items in an image or video stream. Because this task requires significant computational power, GPU acceleration optimization methods are used to achieve real-time performance.

---

## Why GPU Acceleration?
Artificial intelligence inference is a parallel computation challenge because:

--> Convolutional neural networks (CNNs) carry out matrix computations. Meaning That:
- These operations can be performed simultaneously.
- GPUs process thousands of threads simultaneously.

Also, as it has been known, GPU acceleration significantly improves inference speed compared to CPUs.

---

## Hardware Selection
- NVIDIA Jetson Board (embedded GPU platform)
- USB Camera
- Monitor, Keyboard, and mouse
- Internet Connection

The Jetson Nano was selected by the professor; however, the reason why Jetson Nano is a good fit for this course is due to the fact that it supports CUDA and its suitability for parallel embedded AI applications. 

---

## Software Frameworks & Tools
| Tool | Purpose |
|---|---|
| Jetson Linux / JetPack | Operating system and NVIDIA software support |
| Docker / jetson-containers | Containerized environment for compatible PyTorch and CUDA setup |
| PyTorch | Deep learning framework |
| CUDA | GPU acceleration |
| Ultralytics YOLOv8 | Object detection model and inference framework |
| OpenCV | Camera capture and image display |
| Python 3 | Main programming language |

---

## Pre-trained Model
The selected model is YOLOv8n because it is lightweight and suitable for real-time embedded applications, making it good for small devices where power, memory, and processing resources are limited.
Also, it is good to mention that:

- It supports real-time object detection.
- It is easy to use with the Ultralytics Python library.
- It can run on a GPU using PyTorch and CUDA.
- It is suitable for live camera-based applications.

*Beforehand, the ObjectDetection Pre-trained model was going to be used, but due to compatibility issues and complexity coming from the Tensorflow framework, it was decided to move on and work with an easier framework, as Yolov8 and PyTorch offer.
