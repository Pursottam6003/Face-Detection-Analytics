import os
import time
import cv2
from huggingface_hub import hf_hub_download

from analytics import SimpleFaceTracker, VideoAnalytics



# ============================================================
# Configuration
# ============================================================

INPUT_VIDEO = "test_video.mp4"
OUTPUT_VIDEO = "outputs/result_video.mp4"

MODEL_DIR = "models"
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

MODEL_REPO = "opencv/face_detection_yunet"

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000


# ============================================================
# Utility Functions
# ============================================================

def download_model():
    """
    Download the YuNet ONNX model if it is not already
    available locally.
    """

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists: {MODEL_PATH}")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("YuNet model not found.")
    print("Downloading model...")
    print(f"URL: {MODEL_URL}")

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download YuNet model: {e}"
        )

    print(f"Model downloaded successfully: {MODEL_PATH}")

def get_model():
    """
    Returns the path to the YuNet face detection model.

    If the model already exists locally in the models directory,
    it is used directly. Otherwise, the model is downloaded from
    the official OpenCV YuNet repository on Hugging Face.
    """

    # Use the local model if it already exists.
    if os.path.exists(MODEL_PATH):
        print(f"Using local model: {MODEL_PATH}")
        return MODEL_PATH

    print("Local YuNet model not found.")
    print("Downloading model from the official OpenCV repository...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        downloaded_model = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_NAME,
            local_dir=MODEL_DIR
        )

        print(f"Model downloaded successfully: {downloaded_model}")

        return downloaded_model

    except Exception as e:
        raise RuntimeError(
            "Unable to download the YuNet model. "
            "Please check your internet connection or manually "
            f"place {MODEL_NAME} inside the {MODEL_DIR} directory.\n"
            f"Error: {e}"
        )

def create_detector(model_path):
    """
    Create and configure the YuNet face detector.
    """

    return cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        TOP_K
    )


def process_video(detector):
    """
    Process the input video frame-by-frame.

    Pipeline:
        Video Frame
            ↓
        YuNet Face Detection
            ↓
        IoU-based Face Tracking
            ↓
        Analytics Collection
            ↓
        Annotated Output Video

    Analytics generated:
        - Total face detections
        - Unique tracked face IDs
        - Maximum simultaneous faces
        - Average faces visible
        - Face-visible percentage
        - Per-frame face count
        - Per-track visibility duration
        - Charts
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_VIDEO),
        exist_ok=True
    )

    os.makedirs(
        "analytics/charts",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open input video: {INPUT_VIDEO}"
        )

    # --------------------------------------------------------
    # Read video properties
    # --------------------------------------------------------

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    # Some videos may not report a valid FPS.
    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # --------------------------------------------------------
    # Configure YuNet
    # --------------------------------------------------------

    detector.setInputSize(
        (width, height)
    )

    # --------------------------------------------------------
    # Initialize tracker
    # --------------------------------------------------------

    tracker = SimpleFaceTracker(
        iou_threshold=0.3,
        max_missing=10
    )

    # IMPORTANT:
    # Analytics must be initialized AFTER FPS is known.
    analytics = VideoAnalytics(fps)

    # --------------------------------------------------------
    # Create output video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Unable to create output video: {OUTPUT_VIDEO}"
        )

    # --------------------------------------------------------
    # Processing statistics
    # --------------------------------------------------------

    frame_count = 0

    start_time = time.time()

    print("\n" + "=" * 60)
    print("FACE LOCALIZATION + TRACKING")
    print("=" * 60)

    print(f"Resolution    : {width} x {height}")
    print(f"FPS           : {fps:.2f}")
    print(f"Total frames  : {total_frames}")
    print("=" * 60)

    # --------------------------------------------------------
    # Main processing loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Use zero-based frame index internally.
        frame_index = frame_count

        frame_count += 1

        # ----------------------------------------------------
        # Face Detection
        # ----------------------------------------------------

        _, faces = detector.detect(frame)

        detections = []

        if faces is not None:

            for face in faces:

                # YuNet format:
                #
                # [x, y, width, height,
                #  landmarks..., confidence]

                x, y, w, h = face[:4]

                confidence = float(
                    face[-1]
                )

                # ------------------------------------------------
                # Convert bounding box
                # ------------------------------------------------

                x1 = max(
                    0,
                    int(x)
                )

                y1 = max(
                    0,
                    int(y)
                )

                x2 = min(
                    width - 1,
                    int(x + w)
                )

                y2 = min(
                    height - 1,
                    int(y + h)
                )

                # Ignore invalid boxes.
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    {
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        "confidence": confidence
                    }
                )

        # ----------------------------------------------------
        # Face Tracking
        # ----------------------------------------------------

        tracks = tracker.update(
            detections,
            frame_index
        )

        # ----------------------------------------------------
        # Analytics
        # ----------------------------------------------------

        analytics.update(
            frame_index,
            detections,
            tracks
        )

        # ----------------------------------------------------
        # Draw tracked faces
        # ----------------------------------------------------

        for track_id, bbox, confidence in tracks:

            x1, y1, x2, y2 = bbox

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Face ID + confidence
            label = (
                f"Face ID: {track_id} "
                f"| {confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(25, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # Frame-level information
        # ----------------------------------------------------

        timestamp = (
            frame_index / fps
        )

        face_count = len(tracks)

        info = (
            f"Time: {timestamp:.1f}s"
            f" | Faces: {face_count}"
        )

        cv2.putText(
            frame,
            info,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Save processed frame
        # ----------------------------------------------------

        writer.write(frame)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_count % 100 == 0:

            progress = (
                frame_count / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Processed: {frame_count}/"
                f"{total_frames} "
                f"({progress:.1f}%)"
            )

    # --------------------------------------------------------
    # Release resources
    # --------------------------------------------------------

    cap.release()
    writer.release()

    # --------------------------------------------------------
    # Processing performance
    # --------------------------------------------------------

    elapsed_time = (
        time.time() - start_time
    )

    processing_fps = (
        frame_count / elapsed_time
        if elapsed_time > 0
        else 0
    )

    # --------------------------------------------------------
    # Generate analytics
    # --------------------------------------------------------

    analytics.save_summary(
        "analytics/summary.json"
    )

    analytics.save_frame_data(
        "analytics/frame_data.csv"
    )

    analytics.save_track_data(
        "analytics/track_data.csv"
    )

    analytics.generate_charts(
        "analytics/charts"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    summary = analytics.summary()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Frames processed       : "
        f"{frame_count}"
    )

    print(
        f"Processing time        : "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        f"Processing speed       : "
        f"{processing_fps:.2f} FPS"
    )

    print(
        f"Total face detections  : "
        f"{summary['total_face_detections']}"
    )

    print(
        f"Unique face tracks     : "
        f"{summary['unique_tracked_faces']}"
    )

    print(
        f"Max simultaneous faces : "
        f"{summary['max_simultaneous_faces']}"
    )

    print(
        f"Average faces visible  : "
        f"{summary['average_faces_visible']:.2f}"
    )

    print(
        f"Face visibility        : "
        f"{summary['face_visibility_percentage']:.2f}%"
    )

    print("\nOutput files:")
    print(
        f"  Video   : {OUTPUT_VIDEO}"
    )

    print(
        "  Summary : analytics/summary.json"
    )

    print(
        "  Frames  : analytics/frame_data.csv"
    )

    print(
        "  Tracks  : analytics/track_data.csv"
    )

    print(
        "  Charts  : analytics/charts/"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 55)
    print("Face Localization using OpenCV + YuNet")
    print("=" * 55)

    # Step 1: Get the local model or download it if unavailable.
    model_path = get_model()

    # Step 2: Initialize face detector.
    detector = create_detector(model_path)

    # Step 3: Process input video.
    process_video(detector)
    
    # ============================================================
    # Generate analytics
    # ============================================================

    analytics_dir = "analytics"

    os.makedirs(
        analytics_dir,
        exist_ok=True
    )

    analytics.save_summary(
        "analytics/summary.json"
    )

    analytics.save_frame_data(
        "analytics/frame_data.csv"
    )

    analytics.save_track_data(
        "analytics/track_data.csv"
    )

    analytics.generate_charts(
        "analytics/charts"
    )

    print()
    print("=" * 55)
    print("VIDEO ANALYTICS")
    print("=" * 55)

    summary = analytics.summary()

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print()
    print("Analytics generated successfully.")
    print("Analytics directory:", analytics_dir)


if __name__ == "__main__":
    main()