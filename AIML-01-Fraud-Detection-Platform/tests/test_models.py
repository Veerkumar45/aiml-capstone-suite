"""
Unit tests for XGBoost Fraud Model and PyG Fraud Ring Graph Detector.
"""

import pytest
import pandas as pd
from models.xgboost_fraud_model import XGBoostFraudModel
from models.gnn_fraud_ring import FraudRingGraphDetector
from models.hybrid_ensemble import HybridFraudEnsemble

def test_gnn_fraud_ring_scoring():
    detector = FraudRingGraphDetector()
    tx_stream = [
        {"user_id": "U1", "device_id": "D_SHARED", "card_id": "C1", "is_fraud": 1},
        {"user_id": "U2", "device_id": "D_SHARED", "card_id": "C2", "is_fraud": 0},
        {"user_id": "U3", "device_id": "D_SHARED", "card_id": "C3", "is_fraud": 0},
        {"user_id": "U4", "device_id": "D_SHARED", "card_id": "C4", "is_fraud": 0},
    ]
    detector.update_graph(tx_stream)
    
    score = detector.compute_ring_score("U5", "D_SHARED", "C1")
    assert score["shared_device_degree"] == 4
    assert score["is_suspected_fraud_ring"] is True
    assert score["ring_risk_score"] > 0.5

def test_hybrid_ensemble():
    ensemble = HybridFraudEnsemble(xgb_weight=0.6, gnn_weight=0.4)
    xgb_score = 0.90
    gnn_signals = {"ring_risk_score": 0.85, "is_suspected_fraud_ring": True}
    res = ensemble.blend(xgb_score, gnn_signals)
    assert res["risk_tier"] == "CRITICAL"
    assert res["recommended_action"] == "AUTO_BLOCK"
    assert res["blended_fraud_score"] >= 0.85
