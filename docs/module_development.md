# User Activity Analysis & Image Processing Module

## 1. Module Overview

The User Activity Analysis & Image Processing module uses a Raspberry Pi
and camera module to observe and analyze relevant user activities,
particularly for elderly and differently-abled users.

## 2. Hardware

- Raspberry Pi 4 Model B
- Raspberry Pi Camera Module
- Camera ribbon cable
- Micro HDMI monitor during setup
- USB power connection

## 3. Main Software

- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Picamera2
- YOLO ONNX

## 4. Image Processing Pipeline

Camera Input
    ↓
Frame Acquisition
    ↓
Image Preprocessing
    ↓
Person Detection
    ↓
Activity Analysis
    ↓
Fall Classification
    ↓
Temporal Fall Confirmation
    ↓
Output / Telemetry

## 5. Activities

The development included analysis of activities such as:

- Standing
- Walking
- Sitting
- Bending
- Running
- Lying
- Falling
- Idle

## 6. Fall Detection

The fall-detection development used multiple features rather than relying
only on a single image.

Features included:

- Person detection
- Body bounding box
- Body angle
- Width/height ratio
- Vertical movement
- Movement history
- AI fall probability
- Temporal confirmation

## 7. V4 Development

V4 introduced movement tracking and temporal fall detection.

States:

SAFE

POSSIBLE FALL

CONFIRMING

FALL DETECTED

## 8. Camera Processing

Camera frames were acquired using Picamera2.

Resolution:

640 × 480

Format:

RGB888

## 9. AI Classification

The development used a Keras-based image classification model.

Input image size:

224 × 224

The person region was cropped and resized before classification.

## 10. Person Detection

Later development moved from the OpenCV HOG person detector toward
YOLO-based person detection using an ONNX model.

## 11. Dataset

The development used fall and normal image datasets.

The dataset was later inspected and cleaned because the original fall
directory contained incorrectly labelled images.

## 12. Raspberry Pi Integration

The system was designed to run locally on Raspberry Pi and process
camera frames in real time.

## 13. Future Integration

The image-processing module can provide processed video and activity
information to a backend/dashboard system.
