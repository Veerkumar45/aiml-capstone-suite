"""
Exploratory Data Analysis (EDA) Module for Financial Fraud Detection.
Analyzes class imbalance, statistical distributions, velocity metrics, and categorical risk rates.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset_generator import generate_fraud_dataset

def perform_eda(df: pd.DataFrame, report_dir: str = None) -> dict:
    total_records = len(df)
    fraud_records = int(df["is_fraud"].sum())
    legit_records = total_records - fraud_records
    fraud_pct = (fraud_records / total_records) * 100.0
    imbalance_ratio = round(legit_records / max(fraud_records, 1), 1)

    # Amount statistics
    amount_stats_fraud = df[df["is_fraud"] == 1]["amount"].describe().to_dict()
    amount_stats_legit = df[df["is_fraud"] == 0]["amount"].describe().to_dict()

    # Channel fraud rates
    channel_breakdown = df.groupby("channel")["is_fraud"].agg(["count", "sum", "mean"]).reset_index()
    channel_breakdown.columns = ["channel", "total_transactions", "fraud_count", "fraud_rate"]
    channel_breakdown["fraud_rate"] = (channel_breakdown["fraud_rate"] * 100).round(2)
    channel_data = channel_breakdown.to_dict(orient="records")

    # Velocity and device flags
    mean_velocity_fraud = float(df[df["is_fraud"] == 1]["tx_velocity_1h"].mean())
    mean_velocity_legit = float(df[df["is_fraud"] == 0]["tx_velocity_1h"].mean())
    device_change_fraud_rate = float(df[df["device_change_flag"] == 1]["is_fraud"].mean() * 100)
    device_nochange_fraud_rate = float(df[df["device_change_flag"] == 0]["is_fraud"].mean() * 100)

    # Fraud ring concentration
    fraud_types = df[df["is_fraud"] == 1]["fraud_type"].value_counts().to_dict()

    summary = {
        "dataset_metrics": {
            "total_transactions": total_records,
            "legitimate_transactions": legit_records,
            "fraud_transactions": fraud_records,
            "fraud_prevalence_pct": round(fraud_pct, 3),
            "class_imbalance_ratio": f"{imbalance_ratio}:1"
        },
        "amount_distributions": {
            "legitimate": {
                "mean": round(amount_stats_legit["mean"], 2),
                "median": round(amount_stats_legit["50%"], 2),
                "std": round(amount_stats_legit["std"], 2),
                "max": round(amount_stats_legit["max"], 2),
            },
            "fraudulent": {
                "mean": round(amount_stats_fraud["mean"], 2),
                "median": round(amount_stats_fraud["50%"], 2),
                "std": round(amount_stats_fraud["std"], 2),
                "max": round(amount_stats_fraud["max"], 2),
            }
        },
        "channel_analysis": channel_data,
        "behavioral_signals": {
            "mean_velocity_1h_fraud": round(mean_velocity_fraud, 2),
            "mean_velocity_1h_legit": round(mean_velocity_legit, 2),
            "device_change_fraud_rate_pct": round(device_change_fraud_rate, 2),
            "device_steady_fraud_rate_pct": round(device_nochange_fraud_rate, 2),
        },
        "fraud_typologies": fraud_types
    }

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        # Write JSON report
        json_path = os.path.join(report_dir, "eda_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Write Markdown Report
        md_path = os.path.join(report_dir, "eda_report.md")
        with open(md_path, "w") as f:
            f.write(generate_markdown_report(summary))

        print(f"[EDA] Summary saved to {json_path} and {md_path}")

    return summary

def generate_markdown_report(summary: dict) -> str:
    m = summary["dataset_metrics"]
    a = summary["amount_distributions"]
    b = summary["behavioral_signals"]

    md = f"""# Exploratory Data Analysis: Financial Fraud Detection Dataset

## 1. Executive Summary & Class Imbalance Analysis
- **Total Transactions Analyzed:** {m['total_transactions']:,}
- **Legitimate Transactions:** {m['legitimate_transactions']:,} ({100 - m['fraud_prevalence_pct']:.2f}%)
- **Fraudulent Transactions:** {m['fraud_transactions']:,} (**{m['fraud_prevalence_pct']:.2f}%**)
- **Imbalance Ratio:** **{m['class_imbalance_ratio']}** (Extreme Imbalance)

> **Key Observation:** Because fraud comprises less than 2% of total transactions, standard evaluation metrics such as classification Accuracy are deceptive (a trivial zero-classifier yields >98% accuracy while failing entirely). Training requires class-weighted loss (`scale_pos_weight`) and evaluation must use **Precision-Recall AUC (PR-AUC)**.

## 2. Transaction Amount Dynamics
| Metric | Legitimate (Class 0) | Fraudulent (Class 1) | Ratio |
|---|---|---|---|
| **Mean Amount** | ${a['legitimate']['mean']:.2f} | ${a['fraudulent']['mean']:.2f} | {a['fraudulent']['mean'] / max(a['legitimate']['mean'], 0.01):.1f}x |
| **Median Amount** | ${a['legitimate']['median']:.2f} | ${a['fraudulent']['median']:.2f} | {a['fraudulent']['median'] / max(a['legitimate']['median'], 0.01):.1f}x |
| **Max Amount** | ${a['legitimate']['max']:.2f} | ${a['fraudulent']['max']:.2f} | - |

Fraudulent transactions systematically diverge from normal historical user baselines, showing a heavy right-tail distribution typical of account takeovers and fund extraction.

## 3. Behavioral and Velocity Indicators
- **1-Hour Transaction Velocity:** Fraudsters exhibit an average velocity of **{b['mean_velocity_1h_fraud']} tx/hr** versus **{b['mean_velocity_1h_legit']} tx/hr** for legitimate cardholders.
- **Device Change Correlation:** Transactions accompanied by a device change had a fraud rate of **{b['device_change_fraud_rate_pct']}%**, compared to **{b['device_steady_fraud_rate_pct']}%** when staying on recognized hardware.

## 4. Channel Risk Breakdown
| Channel | Total Volume | Fraud Count | Fraud Rate (%) |
|---|---|---|---|
"""
    for ch in summary["channel_analysis"]:
        md += f"| {ch['channel']} | {ch['total_transactions']:,} | {ch['fraud_count']} | {ch['fraud_rate']}% |\n"

    md += """
## 5. Modeling Implications
1. **Model Objective:** Use XGBoost with `scale_pos_weight = Imbalance Ratio` to heavily penalize false negatives.
2. **Feature Store Integration:** Real-time ingestion requires millisecond access to historical rolling averages and transaction velocity windows (via Feast).
3. **Graph Topology:** Account collusion and device sharing rings require Graph Neural Networks (PyG) as tabular models cannot effectively capture high-degree multi-hop node sharing.
"""
    return md

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "transactions_sample.csv")
    if not os.path.exists(data_path):
        df = generate_fraud_dataset(n_transactions=12000, output_path=data_path)
    else:
        df = pd.read_csv(data_path)

    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    res = perform_eda(df, report_dir=report_dir)
    print(f"EDA Completed: {res['dataset_metrics']['fraud_prevalence_pct']}% fraud rate detected.")
