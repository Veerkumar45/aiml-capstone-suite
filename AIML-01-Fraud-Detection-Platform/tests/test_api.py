"""
Unit and integration tests for FastAPI Scoring endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from services.api import app

client = TestClient(app)

def test_health_endpoint():
    with TestClient(app) as tc:
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "HEALTHY"
        assert "XGBoost_Classifier_v1" in data["models_loaded"]

def test_predict_scoring():
    with TestClient(app) as tc:
        payload = {
            "transaction_id": "TX_TEST_001",
            "user_id": "USR_TEST_01",
            "card_id": "CARD_TEST_01",
            "device_id": "DEV_TEST_01",
            "amount": 2500.0,
            "channel": "ONLINE_TRANSFER",
            "location": "New York"
        }
        resp = tc.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "blended_fraud_score" in data
        assert "recommended_action" in data
        assert "inference_latency_ms" in data
        assert data["inference_latency_ms"] < 200.0

def test_feedback_loop():
    with TestClient(app) as tc:
        payload = {
            "transaction_id": "TX_TEST_001",
            "analyst_id": "ANALYST_42",
            "analyst_decision": "CONFIRMED_FRAUD",
            "notes": "Verified unauthorized SIM swap"
        }
        resp = tc.post("/feedback", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "SUCCESS"
