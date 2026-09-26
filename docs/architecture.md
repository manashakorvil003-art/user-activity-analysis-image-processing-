# System Architecture

## Overview

The User Activity Analysis & Image Processing module processes camera frames on a Raspberry Pi and applies computer-vision and machine-learning techniques for human activity and fall analysis.

## Hardware

- Raspberry Pi 4 Model B
- Raspberry Pi Camera Module
- Micro HDMI monitor for setup and testing
- USB power connection

## Software

- Raspberry Pi OS
- Python
- OpenCV
- TensorFlow / Keras
- Picamera2
- OpenCV DNN
- YOLO11n ONNX model

## Processing Architecture

```text
┌──────────────────────────┐
│ Raspberry Pi Camera      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Frame Capture            │
│ Picamera2                │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Image Processing         │
│ OpenCV                   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Person Detection         │
│ YOLO11n ONNX             │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Person Bounding Box      │
│ and Image Crop           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Image Preprocessing      │
│ Resize: 224 × 224        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Fall Classification      │
│ MobileNetV2              │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Fall / Normal Result     │
└──────────────────────────┘
