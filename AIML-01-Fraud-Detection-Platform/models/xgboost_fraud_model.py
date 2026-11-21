"""
XGBoost Fraud Detection Model Pipeline.
Includes:
- Imbalanced learning adjustments (scale_pos_weight / focal weighting)
- Precision-Recall AUC (PR-AUC) optimization
- MLflow run tracking and model checkpointing
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Optional MLflow tracking integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Try XGBoost, fallback to scikit-learn GradientBoosting if not installed
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGBOOST = False

class XGBoostFraudModel:
    FEATURE_COLS = [
        "amount",
        "user_mean_amount",
        "amount_to_mean_ratio",
        "tx_velocity_1h",
        "device_change_flag"
    ]
    CATEGORICAL_COLS = ["channel"]

    def __init__(self, scale_pos_weight: float = 65.0, max_depth: int = 5, learning_rate: float = 0.05):
        self.scale_pos_weight = scale_pos_weight
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.channel_dummies = ["channel_ATM_WITHDRAWAL", "channel_ONLINE_COMMERCE", "channel_ONLINE_TRANSFER", "channel_P2P_TRANSFER", "channel_POS", "channel_WIRE_TRANSFER"]

    def _preprocess(self, df: pd.DataFrame, fit_dummies: bool = False) -> pd.DataFrame:
        X = df[self.FEATURE_COLS].copy()
        dummies = pd.get_dummies(df["channel"], prefix="channel")
        for col in self.channel_dummies:
            if col in dummies.columns:
                X[col] = dummies[col].astype(int)
            else:
                X[col] = 0
        return X

    def train(self, df: pd.DataFrame, experiment_name: str = "Fraud_Detection_XGBoost") -> Dict[str, Any]:
        print(f"[Model Training] Preparing {len(df)} samples...")
        X = self._preprocess(df)
        y = df["is_fraud"].values

        imbalance_ratio = (len(y) - sum(y)) / max(sum(y), 1)
        self.scale_pos_weight = float(round(imbalance_ratio, 2))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        if HAS_XGBOOST:
            print(f"[Model Training] Training XGBClassifier with scale_pos_weight={self.scale_pos_weight}...")
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        else:
            print(f"[Model Training] Using HistGradientBoostingClassifier with class weights...")
            self.model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                class_weight="balanced",
                random_state=42
            )
            self.model.fit(X_train, y_train)

        # Probabilities & Evaluation
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        pr_auc = float(average_precision_score(y_test, y_prob))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "fraud_precision": round(report["1"]["precision"], 4),
            "fraud_recall": round(report["1"]["recall"], 4),
            "fraud_f1": round(report["1"]["f1-score"], 4),
            "confusion_matrix": cm,
            "test_samples": len(y_test),
            "actual_frauds_in_test": int(sum(y_test))
        }

        print(f"[Model Evaluation] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Recall: {metrics['fraud_recall']:.4f}")

        # Track in MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(experiment_name)
                with mlflow.start_run(run_name="xgboost_scale_pos_weight"):
                    mlflow.log_params({
                        "algorithm": "XGBoost" if HAS_XGBOOST else "HistGradientBoosting",
                        "scale_pos_weight": self.scale_pos_weight,
                        "max_depth": self.max_depth,
                        "learning_rate": self.learning_rate
                    })
                    mlflow.log_metrics({
                        "pr_auc": metrics["pr_auc"],
                        "roc_auc": metrics["roc_auc"],
                        "fraud_precision": metrics["fraud_precision"],
                        "fraud_recall": metrics["fraud_recall"],
                        "fraud_f1": metrics["fraud_f1"]
                    })
                    print("[MLflow] Logged run parameters and evaluation metrics successfully.")
            except Exception as e:
                print(f"[MLflow] Run logging skipped: {e}")

        return metrics

    def predict_risk(self, feature_dict: Dict[str, Any]) -> float:
        """
        Predicts fraud risk probability [0.0, 1.0] for a single real-time feature vector.
        """
        row = {
            "amount": feature_dict.get("amount", 50.0),
            "user_mean_amount": feature_dict.get("user_mean_amount", 50.0),
            "amount_to_mean_ratio": feature_dict.get("amount_to_mean_ratio", 1.0),
            "tx_velocity_1h": feature_dict.get("tx_velocity_1h", 1.0),
            "device_change_flag": feature_dict.get("device_change_flag", 0),
            "channel": feature_dict.get("channel", "ONLINE_COMMERCE")
        }
        df_single = pd.DataFrame([row])
        X = self._preprocess(df_single)
        if self.model is None:
            # Cold-start heuristic score
            score = 0.1
            if row["amount_to_mean_ratio"] > 4.0: score += 0.4
            if row["device_change_flag"] == 1: score += 0.25
            if row["tx_velocity_1h"] > 5.0: score += 0.2
            return min(round(score, 4), 0.99)
        prob = float(self.model.predict_proba(X)[0, 1])
        return round(prob, 4)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Model Saved] Checkpoint written to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "XGBoostFraudModel":
        return joblib.load(filepath)

if __name__ == "__main__":
    from data.dataset_generator import generate_fraud_dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "transactions_sample.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_fraud_dataset(n_transactions=5000, output_path=data_path)

    trainer = XGBoostFraudModel()
    metrics = trainer.train(df)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "xgboost_model.joblib")
    trainer.save(save_path)
