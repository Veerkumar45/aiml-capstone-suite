"""
Comprehensive Automated Test Suite for AIML-01 and AIML-04.
Uses Python standard library unittest for zero-dependency execution.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "AIML-01-Fraud-Detection-Platform"))
sys.path.insert(0, os.path.join(BASE_DIR, "AIML-04-Autonomous-Driving-Perception"))

class TestAIML01FraudPlatform(unittest.TestCase):
    def test_01_feature_store_materialization(self):
        from features.feature_store import FraudFeatureStore
        db_path = os.path.join(BASE_DIR, "AIML-01-Fraud-Detection-Platform", "features", "feast_repo", "data", "test_store.db")
        store = FraudFeatureStore(db_path=db_path)
        sample_df = pd.DataFrame([
            {"user_id": "U101", "amount": 50.0, "device_id": "D1", "location": "NYC", "timestamp": "2026-10-01T10:00:00"},
            {"user_id": "U101", "amount": 150.0, "device_id": "D1", "location": "NYC", "timestamp": "2026-10-01T11:00:00"}
        ])
        store.materialize_from_dataframe(sample_df)
        feats = store.get_online_features("U101", "D2", 500.0, "London")
        self.assertEqual(feats["user_id"], "U101")
        self.assertEqual(feats["device_change_flag"], 1)
        self.assertEqual(feats["location_change_flag"], 1)
        self.assertAlmostEqual(feats["amount_to_mean_ratio"], 5.0, places=1)

    def test_02_gnn_fraud_ring_detection(self):
        from models.gnn_fraud_ring import FraudRingGraphDetector
        detector = FraudRingGraphDetector()
        tx_stream = [
            {"user_id": "U1", "device_id": "D_RING", "card_id": "C1", "is_fraud": 1},
            {"user_id": "U2", "device_id": "D_RING", "card_id": "C2", "is_fraud": 0},
            {"user_id": "U3", "device_id": "D_RING", "card_id": "C3", "is_fraud": 0},
            {"user_id": "U4", "device_id": "D_RING", "card_id": "C4", "is_fraud": 0},
        ]
        detector.update_graph(tx_stream)
        score = detector.compute_ring_score("U5", "D_RING", "C1")
        self.assertEqual(score["shared_device_degree"], 4)
        self.assertTrue(score["is_suspected_fraud_ring"])
        self.assertGreater(score["ring_risk_score"], 0.5)

    def test_03_hybrid_ensemble_blending(self):
        from models.hybrid_ensemble import HybridFraudEnsemble
        ensemble = HybridFraudEnsemble(xgb_weight=0.65, gnn_weight=0.35)
        xgb_score = 0.88
        gnn_signals = {"ring_risk_score": 0.90, "is_suspected_fraud_ring": True}
        res = ensemble.blend(xgb_score, gnn_signals)
        self.assertEqual(res["risk_tier"], "CRITICAL")
        self.assertEqual(res["recommended_action"], "AUTO_BLOCK")
        self.assertGreaterEqual(res["blended_fraud_score"], 0.85)

    def test_04_xgboost_training_and_evaluation(self):
        from models.xgboost_fraud_model import XGBoostFraudModel
        from data.dataset_generator import generate_fraud_dataset
        df = generate_fraud_dataset(n_transactions=500, n_users=50)
        model = XGBoostFraudModel()
        metrics = model.train(df)
        self.assertIn("pr_auc", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreater(metrics["pr_auc"], 0.4)

class TestAIML04PerceptionStack(unittest.TestCase):
    def test_01_tracking_and_iou(self):
        from tracking.tracker import ByteTracker, compute_iou
        b1 = [0, 0, 10, 10]
        b2 = [0, 0, 10, 10]
        self.assertAlmostEqual(compute_iou(b1, b2), 1.0, places=2)

        tracker = ByteTracker(iou_threshold=0.3)
        dets1 = [{"bbox": [50.0, 50.0, 100.0, 100.0], "class_name": "car", "confidence": 0.95}]
        t1 = tracker.update(dets1)
        self.assertEqual(len(t1), 1)
        tid = t1[0]["track_id"]

        dets2 = [{"bbox": [53.0, 52.0, 103.0, 102.0], "class_name": "car", "confidence": 0.94}]
        t2 = tracker.update(dets2)
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0]["track_id"], tid)

    def test_02_classical_lane_detector(self):
        from lanes.classical_lane_detector import ClassicalLaneDetector
        detector = ClassicalLaneDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lanes = detector.detect_lanes(frame)
        self.assertIn("curvature_radius_m", lanes)
        self.assertIn("center_offset_m", lanes)
        self.assertIn("lane_status", lanes)

    def test_03_depth_and_collision_warning(self):
        from depth.midas_depth_estimator import MiDaSDepthEstimator
        estimator = MiDaSDepthEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dmap = estimator.estimate_depth_map(frame)
        self.assertEqual(dmap.shape, (480, 640))

        # Close obstacle (< 10m) should trigger emergency brake
        dets = [{"bbox": [200, 350, 440, 470], "class_name": "car", "confidence": 0.92}]
        res = estimator.compute_obstacle_distances(dmap, dets)
        self.assertEqual(len(res), 1)
        self.assertIn("distance_m", res[0])
        self.assertIn("collision_alert", res[0])

    def test_04_full_perception_pipeline(self):
        from ros2_carla.ros2_perception_node import AutonomousPerceptionPipeline
        from data.dataset_loader import PerceptionDatasetLoader
        pipeline = AutonomousPerceptionPipeline()
        frame, _ = PerceptionDatasetLoader.generate_synthetic_frame(0)
        out = pipeline.process_frame(frame)
        self.assertIn("fps", out)
        self.assertIn("latency_ms", out)
        self.assertIn("tracked_objects", out)
        self.assertIn("lane_metrics", out)
        self.assertIn("emergency_brake_triggered", out)

if __name__ == "__main__":
    unittest.main(verbosity=2)
