# Testing

## Overview

The User Activity Analysis & Image Processing module was tested using dataset images and Raspberry Pi camera processing.

Testing focused on:

- Person detection
- Image classification
- Fall detection
- Real-time processing
- Model predictions
- Temporal fall confirmation

## Dataset Testing

The original binary dataset contained:

| Class | Images |
|---|---:|
| Fall | 115 |
| Normal | 875 |
| Total | 990 |

During auditing, label-quality problems were identified in the original fall directory.

A clean subset containing 41 genuine fall images was created for further dataset improvement.

## Model Testing

### V2 Model

The V2 model was evaluated using the available test split.

The initial evaluation produced very high accuracy.

However, the original dataset contained incorrectly labelled images. Therefore, the evaluation result may not represent real-world performance.

### V3 Model

The V3 model produced:

| Metric | Result |
|---|---:|
| Fall Recall | 100% |
| Fall Precision | 35% |
| Normal Recall | 90.15% |

The fall precision result shows that false-positive fall predictions remained an issue.

## Real-Time Testing

The processing pipeline was also tested using Raspberry Pi camera input.

The system was evaluated for:

- Person detection
- Bounding-box tracking
- Fall probability
- Body geometry
- Temporal movement
- Fall-state transitions
- Frames per second

## Fall Detection Testing

The V4 implementation uses multiple signals instead of relying only on a single frame.

```text
Person Detection
       ↓
Body Geometry
       ↓
Vertical Movement
       ↓
AI Classification
       ↓
Temporal Confirmation
       ↓
Fall State
