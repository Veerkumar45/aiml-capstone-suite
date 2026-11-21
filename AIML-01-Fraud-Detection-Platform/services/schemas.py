"""
Pydantic Schemas for Financial Fraud Detection API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TX_00098214")
    user_id: str = Field(..., example="USR_00142")
    card_id: str = Field(..., example="CARD_00142")
    device_id: str = Field(..., example="DEV_00891")
    amount: float = Field(..., gt=0, example=1250.00)
    channel: str = Field(default="ONLINE_COMMERCE", example="ONLINE_COMMERCE")
    location: str = Field(default="New York", example="New York")
    ip_address: Optional[str] = Field(default="192.168.1.10", example="192.168.1.10")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())

class FraudScoreResponse(BaseModel):
    transaction_id: str
    user_id: str
    blended_fraud_score: float
    risk_tier: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str  # APPROVE, STEP_UP_2FA, MANUAL_REVIEW, AUTO_BLOCK
    inference_latency_ms: float
    feature_signals: Dict[str, Any]
    model_breakdown: Dict[str, Any]
    graph_signals: Dict[str, Any]

class AnalystFeedbackRequest(BaseModel):
    transaction_id: str
    analyst_id: str
    analyst_decision: str  # CONFIRMED_FRAUD, FALSE_POSITIVE, APPROVED
    notes: Optional[str] = ""
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]
    feature_store_connected: bool
