"""
Unit tests for MiDaS monocular depth estimation and collision safety.
"""

import pytest
import numpy as np
from depth.midas_depth_estimator import MiDaSDepthEstimator

def test_depth_estimation_and_metric_distance():
    estimator = MiDaSDepthEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dmap = estimator.estimate_depth_map(dummy_frame)
    assert dmap.shape == (480, 640)
    assert dmap.min() >= 0.0
    assert dmap.max() <= 1.0

    detections = [
        {"bbox": [200, 300, 300, 400], "class_name": "car", "confidence": 0.9}
    ]
    res = estimator.compute_obstacle_distances(dmap, detections)
    assert len(res) == 1
    assert "distance_m" in res[0]
    assert "collision_alert" in res[0]
    assert res[0]["distance_m"] > 0
