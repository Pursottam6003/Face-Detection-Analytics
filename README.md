# Face Localization in Video

## Overview

This project detects and localizes faces of characters in a video and draws
bounding boxes around the detected faces.

The pipeline also includes lightweight face tracking and analytics to
understand face presence throughout the video.

---

## Approach

| Component | Technology |
|---|---|
| Video Processing | OpenCV |
| Face Detection | YuNet |
| Model Format | ONNX |
| Tracking | Lightweight IoU-based Tracker |
| Analytics | Python, Pandas, Matplotlib |

---

## Pipeline

```text
                    Input Video
                         │
                         ▼
                YuNet Face Detection
                         │
                         ▼
                  Bounding Boxes
                         │
                         ▼
               Lightweight IoU Tracker
                         │
                         ▼
                     Track IDs
                         │
                         ▼
                     Analytics
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Face Counts     Track Duration   Visibility
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                       Outputs
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   Annotated Video   CSV / JSON       Charts