# User Activity Analysis & Image Processing

A Raspberry Pi-based computer-vision module for analyzing human activity and detecting potential falls using camera-based image processing, machine learning, and temporal analysis.

## Project Overview

This project implements the **User Activity Analysis & Image Processing** module using a Raspberry Pi and camera module.

The system captures video frames, detects people, processes the detected person region, and applies machine-learning and computer-vision techniques for activity and fall analysis.

## Key Features

- Real-time camera frame capture
- Human/person detection
- Image preprocessing and enhancement
- Fall and normal activity classification
- Body geometry analysis
- Temporal movement analysis
- Fall-state confirmation
- Raspberry Pi deployment
- TensorFlow/Keras-based classification
- YOLO11n-based person detection in later versions

## Hardware

- Raspberry Pi 4 Model B
- Raspberry Pi Camera Module
- Micro HDMI monitor for setup and testing
- USB power connection

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing |
| TensorFlow / Keras | Machine learning |
| MobileNetV2 | Image classification |
| YOLO11n | Person detection |
| Picamera2 | Raspberry Pi camera interface |
| ONNX | YOLO model format |

## System Architecture

```text
Camera
   │
   ▼
Frame Capture
   │
   ▼
Image Processing
   │
   ▼
Person Detection
   │
   ▼
Person Bounding Box
   │
   ▼
Image Preprocessing
   │
   ▼
AI Classification
   │
   ▼
Fall / Normal
   │
   ▼
Temporal Analysis
   │
   ▼
Fall Detection Result
