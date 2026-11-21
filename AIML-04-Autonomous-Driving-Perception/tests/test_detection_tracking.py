"""
Unit tests for YOLO detection and ByteTrack tracking.
"""

import pytest
import numpy as np
from detection.yolo_detector import YOLODetector
from tracking.tracker import ByteTracker, compute_iou

def test_iou_calculation():
    b1 = [0, 0, 10, 10]
    b2 = [0, 0, 10, 10]
    assert compute_iou(b1, b2) == pytest.approx(1.0, 0.01)

    b3 = [10, 10, 20, 20]
    assert compute_iou(b1, b3) == pytest.approx(0.0, 0.01)

def test_tracker_association():
    tracker = ByteTracker(iou_threshold=0.3)
    frame1_dets = [
        {"bbox": [50.0, 50.0, 100.0, 100.0], "class_name": "car", "confidence": 0.95}
    ]
    t1 = tracker.update(frame1_dets)
    assert len(t1) == 1
    tid = t1[0]["track_id"]

    # Frame 2: slightly shifted
    frame2_dets = [
        {"bbox": [54.0, 52.0, 104.0, 102.0], "class_name": "car", "confidence": 0.94}
    ]
    t2 = tracker.update(frame2_dets)
    assert len(t2) == 1
    assert t2[0]["track_id"] == tid
