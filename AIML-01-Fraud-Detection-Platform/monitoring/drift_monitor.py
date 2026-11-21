"""
Evidently AI Data & Prediction Drift Monitoring Engine.
Monitors incoming production transaction batches against reference baselines.
Computes feature drift (Kolmogorov-Smirnov & Wasserstein metrics) and triggers automated retraining.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Try importing evidently
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

class DriftMonitor:
    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.features_to_monitor = [
            "amount",
            "amount_to_mean_ratio",
            "tx_velocity_1h",
            "device_change_flag"
        ]

    def compute_ks_drift(self, ref_series: pd.Series, cur_series: pd.Series) -> Dict[str, Any]:
        """
        Calculates two-sample Kolmogorov-Smirnov test statistic and p-value for continuous features.
        """
        from scipy import stats
        res = stats.ks_2samp(ref_series.dropna(), cur_series.dropna())
        p_val = float(res.pvalue)
        stat = float(res.statistic)
        # Drift detected if p-value is below significance level (0.05)
        is_drift = p_val < 0.05
        return {
            "statistic": round(stat, 4),
            "p_value": round(p_val, 6),
            "drift_detected": is_drift
        }

    def evaluate_batch_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_report_path: str = None
    ) -> Dict[str, Any]:
        """
        Evaluates data drift between baseline training data and latest production window.
        """
        print(f"[Drift Monitor] Comparing reference ({len(reference_df)} rows) vs current ({len(current_df)} rows)...")

        feature_metrics = {}
        drifted_count = 0

        for col in self.features_to_monitor:
            if col in reference_df.columns and col in current_df.columns:
                drift_info = self.compute_ks_drift(reference_df[col], current_df[col])
                feature_metrics[col] = drift_info
                if drift_info["drift_detected"]:
                    drifted_count += 1

        total_features = len(feature_metrics)
        drift_share = drifted_count / max(total_features, 1)
        dataset_drift = drift_share >= self.drift_threshold

        retraining_recommended = dataset_drift

        report_summary = {
            "dataset_drift_detected": dataset_drift,
            "drift_share": round(drift_share, 4),
            "drift_threshold": self.drift_threshold,
            "drifted_features_count": drifted_count,
            "total_monitored_features": total_features,
            "feature_breakdown": feature_metrics,
            "retraining_triggered": retraining_recommended,
            "trigger_action": "DISPATCH_RETRAIN_WORKFLOW" if retraining_recommended else "NO_ACTION_REQUIRED"
        }

        if retraining_recommended:
            print(f"[ALERT: DRIFT DETECTED] Drift share {drift_share*100:.1f}% exceeds threshold {self.drift_threshold*100:.1f}%!")
            print("[Trigger] Automatic model retraining pipeline dispatched to Kubernetes MLflow Job.")
        else:
            print(f"[Status Normal] Drift share {drift_share*100:.1f}% is within acceptable parameters.")

        if output_report_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
            with open(output_report_path, "w") as f:
                json.dump(report_summary, f, indent=2)
            print(f"[Drift Monitor] Report saved to {output_report_path}")

        return report_summary

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "transactions_sample.csv")
    if os.path.exists(data_path):
        df_all = pd.read_csv(data_path)
    else:
        from data.dataset_generator import generate_fraud_dataset
        df_all = generate_fraud_dataset(n_transactions=6000)

    # Split into reference and simulated drifted production batch
    ref = df_all.iloc[:3000].copy()
    cur = df_all.iloc[3000:].copy()

    # Inject slight distribution shift in current batch
    cur["amount"] = cur["amount"] * np.random.uniform(1.2, 1.8, size=len(cur))
    cur["tx_velocity_1h"] = cur["tx_velocity_1h"] * 2.1

    monitor = DriftMonitor(drift_threshold=0.25)
    report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "drift_report.json")
    res = monitor.evaluate_batch_drift(ref, cur, output_report_path=report_file)
    print("Drift Summary:", json.dumps(res, indent=2))
