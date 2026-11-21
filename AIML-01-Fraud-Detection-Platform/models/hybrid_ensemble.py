"""
Hybrid Risk Scoring Ensemble.
Blends:
1. XGBoost Score: Individual behavioral anomaly (velocity, amount-to-mean, device change)
2. PyG GNN Score: Network topological collusion (shared device clusters, fraud rings)
"""

import os
from typing import Dict, Any

class HybridFraudEnsemble:
    def __init__(self, xgb_weight: float = 0.65, gnn_weight: float = 0.35):
        self.xgb_weight = xgb_weight
        self.gnn_weight = gnn_weight

    def blend(self, xgb_score: float, gnn_dict: Dict[str, Any]) -> Dict[str, Any]:
        gnn_score = gnn_dict.get("ring_risk_score", 0.05)
        is_suspected_ring = gnn_dict.get("is_suspected_fraud_ring", False)

        # Weighted combination
        blended_score = (self.xgb_weight * xgb_score) + (self.gnn_weight * gnn_score)

        # Ring override boost: If node is part of a verified dense ring, elevate score floor
        if is_suspected_ring and blended_score < 0.75:
            blended_score = max(blended_score + 0.25, 0.75)

        blended_score = round(min(blended_score, 0.9999), 4)

        # Action classification
        if blended_score >= 0.80:
            action = "AUTO_BLOCK"
            risk_tier = "CRITICAL"
        elif blended_score >= 0.60:
            action = "MANUAL_REVIEW"
            risk_tier = "HIGH"
        elif blended_score >= 0.35:
            action = "STEP_UP_2FA"
            risk_tier = "MEDIUM"
        else:
            action = "APPROVE"
            risk_tier = "LOW"

        return {
            "blended_fraud_score": blended_score,
            "risk_tier": risk_tier,
            "recommended_action": action,
            "model_breakdown": {
                "xgboost_tabular_score": xgb_score,
                "gnn_graph_ring_score": gnn_score,
                "xgb_weight": self.xgb_weight,
                "gnn_weight": self.gnn_weight
            },
            "graph_signals": gnn_dict
        }
