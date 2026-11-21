"""
Multi-Object Tracker (ByteTrack / Kalman Filter Architecture).
Maintains temporal object identities and trajectories across consecutive video frames
for vehicles, pedestrians, and obstacles.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = b1_area + b2_area - inter_area
    return inter_area / max(union_area, 1e-6)

class KalmanTrack:
    def __init__(self, track_id: int, bbox: List[float], class_name: str, confidence: float):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox # [x1, y1, x2, y2]
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = [bbox]

        # Velocity estimation
        self.vx = 0.0
        self.vy = 0.0

    def predict(self):
        # Constant velocity prediction
        self.bbox[0] += self.vx
        self.bbox[1] += self.vy
        self.bbox[2] += self.vx
        self.bbox[3] += self.vy
        self.age += 1
        self.time_since_update += 1

    def update(self, new_bbox: List[float], conf: float):
        # Update velocities based on displacement
        old_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        old_cy = (self.bbox[1] + self.bbox[3]) / 2.0
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0

        self.vx = 0.6 * self.vx + 0.4 * (new_cx - old_cx)
        self.vy = 0.6 * self.vy + 0.4 * (new_cy - old_cy)

        self.bbox = new_bbox
        self.confidence = conf
        self.hits += 1
        self.time_since_update = 0
        self.history.append(new_bbox)
        if len(self.history) > 30:
            self.history.pop(0)

class ByteTracker:
    def __init__(self, max_time_lost: int = 15, iou_threshold: float = 0.3):
        self.max_time_lost = max_time_lost
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanTrack] = []
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 1. Predict track positions
        for t in self.tracks:
            t.predict()

        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        matches = []

        if len(self.tracks) > 0 and len(detections) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = compute_iou(track.bbox, det["bbox"])

            # Greedy Hungarian-style matching
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_iou < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matches.append((t_idx, d_idx))
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

            matched_tracks_set = {m[0] for m in matches}
            matched_dets_set = {m[1] for m in matches}
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks_set]
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets_set]

        # Update matched tracks
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            self.tracks[t_idx].update(det["bbox"], det["confidence"])

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_t = KalmanTrack(self.next_id, det["bbox"], det["class_name"], det["confidence"])
            self.next_id += 1
            self.tracks.append(new_t)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_time_lost]

        # Return active tracks
        results = []
        for t in self.tracks:
            if t.hits >= 1 and t.time_since_update == 0:
                results.append({
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "bbox": [round(c, 1) for c in t.bbox],
                    "confidence": round(t.confidence, 3),
                    "velocity": (round(t.vx, 2), round(t.vy, 2))
                })
        return results

if __name__ == "__main__":
    tracker = ByteTracker()
    d1 = [{"bbox": [100, 100, 150, 150], "class_name": "car", "confidence": 0.9}]
    d2 = [{"bbox": [105, 102, 155, 152], "class_name": "car", "confidence": 0.92}]
    print("Frame 1:", tracker.update(d1))
    print("Frame 2:", tracker.update(d2))
