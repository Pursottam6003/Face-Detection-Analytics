import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Track:
    """
    Represents one tracked face across multiple video frames.

    Important:
    This is a tracking ID, NOT a verified person's identity.
    """

    def __init__(self, track_id, bbox, frame_index):
        self.track_id = track_id
        self.bbox = bbox
        self.last_frame = frame_index

        self.first_frame = frame_index
        self.frames_seen = 1
        self.max_confidence = 0.0

    def update(self, bbox, frame_index, confidence):
        self.bbox = bbox
        self.last_frame = frame_index
        self.frames_seen += 1
        self.max_confidence = max(
            self.max_confidence,
            confidence
        )


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two bounding boxes.

    Box format:
        [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


class SimpleFaceTracker:
    """
    Lightweight IoU-based face tracker.

    This tracker assigns a persistent ID to detections that
    sufficiently overlap with a previous detection.

    This is not facial recognition.
    """

    def __init__(self, iou_threshold=0.3, max_missing=10):

        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_index):
        """
        Update tracks using current frame detections.

        detections:
            list of dictionaries:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float
            }

        Returns:
            list of tuples:
            (track_id, bbox, confidence)
        """

        if not detections:

            # Remove stale tracks
            stale_ids = [
                track_id
                for track_id, track in self.tracks.items()
                if frame_index - track.last_frame > self.max_missing
            ]

            for track_id in stale_ids:
                del self.tracks[track_id]

            return []

        assigned_tracks = set()
        assigned_detections = set()

        matches = []

        # Calculate IoU between every existing track
        # and every current detection.
        for track_id, track in self.tracks.items():

            for detection_index, detection in enumerate(detections):

                if detection_index in assigned_detections:
                    continue

                iou = calculate_iou(
                    track.bbox,
                    detection["bbox"]
                )

                if iou >= self.iou_threshold:

                    matches.append(
                        (
                            iou,
                            track_id,
                            detection_index
                        )
                    )

        # Highest IoU matches first
        matches.sort(reverse=True)

        for iou, track_id, detection_index in matches:

            if track_id in assigned_tracks:
                continue

            if detection_index in assigned_detections:
                continue

            detection = detections[detection_index]

            self.tracks[track_id].update(
                detection["bbox"],
                frame_index,
                detection["confidence"]
            )

            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        # Create new tracks for unmatched detections
        for detection_index, detection in enumerate(detections):

            if detection_index in assigned_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            track = Track(
                track_id,
                detection["bbox"],
                frame_index
            )

            track.max_confidence = detection["confidence"]

            self.tracks[track_id] = track

            assigned_tracks.add(track_id)

        # Remove stale tracks
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if frame_index - track.last_frame > self.max_missing
        ]

        for track_id in stale_ids:
            del self.tracks[track_id]

        results = []

        for track_id in assigned_tracks:

            track = self.tracks[track_id]

            confidence = track.max_confidence

            results.append(
                (
                    track_id,
                    track.bbox,
                    confidence
                )
            )

        return results


class VideoAnalytics:

    def __init__(self, fps):

        self.fps = fps

        self.total_frames = 0
        self.total_face_detections = 0

        self.frames_with_faces = 0
        self.face_count_per_frame = []

        self.track_first_frame = {}
        self.track_last_frame = {}
        self.track_frames = {}

    def update(
        self,
        frame_index,
        detections,
        tracks
    ):

        self.total_frames += 1

        face_count = len(detections)

        self.total_face_detections += face_count

        self.face_count_per_frame.append(
            face_count
        )

        if face_count > 0:
            self.frames_with_faces += 1

        for track_id, bbox, confidence in tracks:

            if track_id not in self.track_first_frame:

                self.track_first_frame[track_id] = frame_index
                self.track_frames[track_id] = 0

            self.track_last_frame[track_id] = frame_index

            self.track_frames[track_id] += 1

    def summary(self):

        duration = (
            self.total_frames / self.fps
            if self.fps > 0
            else 0
        )

        average_faces = (
            np.mean(self.face_count_per_frame)
            if self.face_count_per_frame
            else 0
        )

        max_faces = (
            max(self.face_count_per_frame)
            if self.face_count_per_frame
            else 0
        )

        visibility_percentage = (
            self.frames_with_faces
            / self.total_frames
            * 100
            if self.total_frames > 0
            else 0
        )

        return {
            "video_duration_seconds": round(
                duration,
                2
            ),

            "total_frames": self.total_frames,

            "fps": round(
                self.fps,
                2
            ),

            "total_face_detections":
                self.total_face_detections,

            "unique_tracked_faces":
                len(self.track_first_frame),

            "max_simultaneous_faces":
                max_faces,

            "average_faces_visible":
                round(
                    float(average_faces),
                    2
                ),

            "frames_with_faces":
                self.frames_with_faces,

            "face_visibility_percentage":
                round(
                    visibility_percentage,
                    2
                )
        }

    def save_summary(self, output_path):

        summary = self.summary()

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4
            )

    def save_frame_data(self, output_path):

        rows = []

        for frame_index, face_count in enumerate(
            self.face_count_per_frame
        ):

            rows.append(
                {
                    "frame": frame_index + 1,

                    "timestamp_seconds":
                        round(
                            frame_index / self.fps,
                            3
                        ),

                    "faces_visible":
                        face_count
                }
            )

        df = pd.DataFrame(rows)

        df.to_csv(
            output_path,
            index=False
        )

    def save_track_data(self, output_path):

        rows = []

        for track_id in self.track_first_frame:

            first_frame = self.track_first_frame[
                track_id
            ]

            last_frame = self.track_last_frame[
                track_id
            ]

            frames_seen = self.track_frames[
                track_id
            ]

            rows.append(
                {
                    "track_id":
                        track_id,

                    "first_seen_seconds":
                        round(
                            first_frame / self.fps,
                            2
                        ),

                    "last_seen_seconds":
                        round(
                            last_frame / self.fps,
                            2
                        ),

                    "duration_seconds":
                        round(
                            frames_seen / self.fps,
                            2
                        ),

                    "frames_seen":
                        frames_seen
                }
            )

        df = pd.DataFrame(rows)

        df.to_csv(
            output_path,
            index=False
        )

    def generate_charts(self, output_dir):

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # ----------------------------------------
        # Chart 1: Faces visible over time
        # ----------------------------------------

        timestamps = [
            i / self.fps
            for i in range(
                len(self.face_count_per_frame)
            )
        ]

        plt.figure(figsize=(12, 5))

        plt.plot(
            timestamps,
            self.face_count_per_frame
        )

        plt.title(
            "Faces Visible Over Time"
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Number of Faces"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "faces_over_time.png"
            ),
            dpi=150
        )

        plt.close()

        # ----------------------------------------
        # Chart 2: Track duration
        # ----------------------------------------

        track_ids = list(
            self.track_frames.keys()
        )

        durations = [
            self.track_frames[track_id] / self.fps
            for track_id in track_ids
        ]

        if track_ids:

            plt.figure(figsize=(10, 5))

            plt.bar(
                [
                    f"Track {track_id}"
                    for track_id in track_ids
                ],
                durations
            )

            plt.title(
                "Tracked Face Presence Duration"
            )

            plt.xlabel(
                "Track ID"
            )

            plt.ylabel(
                "Duration (seconds)"
            )

            plt.xticks(
                rotation=45
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    output_dir,
                    "track_duration.png"
                ),
                dpi=150
            )

            plt.close()