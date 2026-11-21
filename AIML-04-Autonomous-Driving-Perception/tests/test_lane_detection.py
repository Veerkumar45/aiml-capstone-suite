"""
Unit tests for Classical and Deep Lane Detection modules.
"""

import pytest
import numpy as np
from lanes.classical_lane_detector import ClassicalLaneDetector
from lanes.segmentation_lane_detector import SegmentationLaneDetector

def test_classical_lane_detector():
    detector = ClassicalLaneDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect_lanes(dummy_frame)
    assert "curvature_radius_m" in res
    assert "center_offset_m" in res
    assert "lane_status" in res
    assert res["curvature_radius_m"] > 0

def test_segmentation_comparison():
    comp = SegmentationLaneDetector.get_architectural_comparison()
    assert "classical_opencv" in comp
    assert "deep_segmentation" in comp
    assert comp["classical_opencv"]["inference_latency_ms"] < comp["deep_segmentation"]["inference_latency_ms"]
