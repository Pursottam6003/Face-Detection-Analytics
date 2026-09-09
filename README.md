# Face Localization in Video

## Overview

This project detects and localizes faces of characters in a video
and draws bounding boxes around detected faces.

## Approach

- Video processing: OpenCV
- Face detection: YuNet
- Model format: ONNX

## Pipeline

Input Video
    ↓
YuNet Face Detection
    ↓
Bounding Boxes
    ↓
Lightweight IoU Tracker
    ↓
Track IDs
    ↓
Analytics
    ├── Unique tracked faces
    ├── Max faces visible simultaneously
    ├── Average faces visible
    ├── Total face detections
    ├── Video duration
    ├── Face-visible duration
    └── Presence duration per track
    ↓
Outputs
    ├── Annotated video
    ├── analytics.csv
    ├── summary.json
    └── charts/
         ├── faces_over_time.png
         └── track_duration.png

## Setup

Create a virtual environment:

python -m venv .venv

Activate the environment:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Run

python main.py

The output will be generated at:

outputs/result_video.mp4

## Model

The application automatically downloads the YuNet model if it is
not available locally.

Model source:
OpenCV Zoo - YuNet

## Parameters

Confidence threshold: 0.6
NMS threshold: 0.3
Top-K: 5000

| Metric                     |    Result |
| -------------------------- | --------: |
| Video duration             | 31.87 sec |
| Frames processed           |       956 |
| Total face detections      |     1,842 |
| Unique face tracks         |         7 |
| Maximum simultaneous faces |         4 |
| Average faces visible      |      1.93 |
| Face-visible percentage    |    84.94% |
 
0s ───────────────────────────── 32s

Faces:
0-5s      ████       2
5-10s     ███████    3
10-15s    ██         1
15-20s    █████      2
20-25s    █████████  4
25-32s    ███        1