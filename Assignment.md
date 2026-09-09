Assignment Questions
Q1. Which video processing library and localization model did you use?

I used OpenCV for video processing and YuNet for face detection and
localization. OpenCV handles frame extraction, bounding-box visualization,
tracking integration, and output video generation, while YuNet predicts face
bounding boxes and confidence scores.

Q2. What would be a better approach with more resources, time, and data?

With additional resources and domain-specific training data, I would use a
more robust detector such as SCRFD or RetinaFace, combined with
ByteTrack or DeepSORT for stable multi-face tracking. Fine-tuning the
detector on the target video domain would further improve performance for
small, occluded, blurred, or difficult-angle faces.

Q3. What are some real-world applications of object detection/localization?

Object detection and localization are widely used in:

    Autonomous Vehicles
    Video surveillance
    Retail analytics
    Healthcare
    Sports analytics
    Industrial automation
    Traffic monitoring


A related project I worked on was a real-time sign language recognition
system, where hand landmarks were detected from video frames and used as
features for gesture classification.

Q4. Briefly explain the ResNet architecture.

ResNet (Residual Network) addresses the degradation and vanishing-gradient
problems encountered when training very deep neural networks.

Instead of learning a complete mapping directly, a residual block learns a
residual function and adds the original input through a skip connection:

Input
  │
  ├──────────────────────┐
  │                      │
  ▼                      │
3×3 Conv → BN → ReLU     │
  │                      │
  ▼                      │
3×3 Conv → BN            │
  │                      │
  └────────── + ◄────────┘
             │
             ▼
           ReLU
             │
           Output


Common ResNet variants include ResNet-18, ResNet-34, ResNet-50,
ResNet-101, and ResNet-152. ResNet-50 and deeper variants use bottleneck
blocks consisting of 1×1 → 3×3 → 1×1 convolutions.

References
OpenCV – YuNet Face Detection
OpenCV – FaceDetectorYN
YuNet ONNX model
OpenCV documentation