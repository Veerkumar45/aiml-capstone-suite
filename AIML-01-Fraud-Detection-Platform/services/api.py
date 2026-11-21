"""
FastAPI Real-Time Fraud Scoring Microservice.
Provides sub-10ms transaction risk evaluation, feature store retrieval,
GNN fraud ring analysis, and analyst feedback loops.
"""

import os
import sys
import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Adjust paths to import siblings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from services.schemas import TransactionRequest, FraudScoreResponse, AnalystFeedbackRequest, HealthResponse
from features.feature_store import FraudFeatureStore
from models.xgboost_fraud_model import XGBoostFraudModel
from models.gnn_fraud_ring import FraudRingGraphDetector
from models.hybrid_ensemble import HybridFraudEnsemble

app = FastAPI(
    title="Real-Time Financial Fraud Detection Platform API",
    description="Sub-10ms transaction scoring with XGBoost, PyG Graph Neural Networks, and Feast Feature Store",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
feature_store: FraudFeatureStore = None
xgb_model: XGBoostFraudModel = None
gnn_detector: FraudRingGraphDetector = None
ensemble: HybridFraudEnsemble = None
feedback_db: List[Dict[str, Any]] = []

@app.on_event("startup")
def startup_event():
    global feature_store, xgb_model, gnn_detector, ensemble
    print("[API Startup] Initializing Feature Store and Models...")
    feature_store = FraudFeatureStore()
    gnn_detector = FraudRingGraphDetector()
    ensemble = HybridFraudEnsemble(xgb_weight=0.65, gnn_weight=0.35)

    # Load or initialize model
    model_path = os.path.join(BASE_DIR, "models", "checkpoints", "xgboost_model.joblib")
    if os.path.exists(model_path):
        try:
            xgb_model = XGBoostFraudModel.load(model_path)
            print("[API Startup] Loaded existing model checkpoint.")
        except Exception:
            xgb_model = XGBoostFraudModel()
    else:
        xgb_model = XGBoostFraudModel()
        print("[API Startup] Initialized XGBoost model.")

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "HEALTHY",
        "version": "1.0.0",
        "models_loaded": ["XGBoost_Classifier_v1", "PyG_Fraud_Ring_GNN_v1"],
        "feature_store_connected": feature_store is not None
    }

@app.post("/predict", response_model=FraudScoreResponse)
def score_transaction(req: TransactionRequest):
    t_start = time.perf_counter()

    # 1. Fetch real-time online features from Feast store
    features = feature_store.get_online_features(
        user_id=req.user_id,
        current_device_id=req.device_id,
        current_amount=req.amount,
        current_city=req.location
    )
    features["channel"] = req.channel
    features["amount"] = req.amount

    # 2. Score with XGBoost tabular model
    xgb_score = xgb_model.predict_risk(features)

    # 3. Update dynamic graph & compute GNN fraud ring score
    gnn_detector.update_graph([{
        "user_id": req.user_id,
        "device_id": req.device_id,
        "card_id": req.card_id,
        "is_fraud": 1 if xgb_score > 0.85 else 0
    }])
    gnn_signals = gnn_detector.compute_ring_score(
        user_id=req.user_id,
        device_id=req.device_id,
        card_id=req.card_id
    )

    # 4. Blend scores through hybrid ensemble
    decision = ensemble.blend(xgb_score, gnn_signals)

    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return {
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "blended_fraud_score": decision["blended_fraud_score"],
        "risk_tier": decision["risk_tier"],
        "recommended_action": decision["recommended_action"],
        "inference_latency_ms": latency_ms,
        "feature_signals": features,
        "model_breakdown": decision["model_breakdown"],
        "graph_signals": decision["graph_signals"]
    }

@app.get("/graph/subgraph/{user_id}")
def get_user_subgraph(user_id: str):
    """
    Returns graph neighborhood for visualization in the analyst dashboard.
    """
    if gnn_detector is None:
        raise HTTPException(status_code=503, detail="Graph engine initializing")
    return gnn_detector.get_subgraph_nodes_and_edges(user_id)

@app.post("/feedback")
def submit_feedback(feedback: AnalystFeedbackRequest):
    """
    Analyst feedback loop: records decision for future retraining and active learning.
    """
    record = feedback.dict()
    feedback_db.append(record)
    print(f"[Feedback Received] TX: {feedback.transaction_id} marked as {feedback.analyst_decision}")
    return {"status": "SUCCESS", "message": "Feedback recorded", "total_feedbacks": len(feedback_db)}

@app.get("/feedback/history")
def get_feedback_history():
    return {"feedbacks": feedback_db}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.api:app", host="0.0.0.0", port=8000, reload=True)
