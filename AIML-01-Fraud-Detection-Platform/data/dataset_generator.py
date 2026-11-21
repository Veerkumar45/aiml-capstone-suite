"""
Financial Fraud Dataset Generator & Preprocessor.
Supports generating synthetic transaction streams following the PaySim and IEEE-CIS schemas,
including realistic transaction velocities, device sharing networks, and fraud rings.
"""

import os
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_fraud_dataset(
    n_transactions: int = 10000,
    n_users: int = 800,
    fraud_rate: float = 0.015,
    random_seed: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generates realistic financial transaction dataset with extreme class imbalance (<2% fraud)
    and synthetic fraud ring topologies (colluding users sharing devices and cards).
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    user_ids = [f"USR_{i:05d}" for i in range(1, n_users + 1)]
    card_ids = [f"CARD_{i:05d}" for i in range(1, int(n_users * 1.2))]
    device_ids = [f"DEV_{i:05d}" for i in range(1, int(n_users * 1.5))]
    ip_addresses = [f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}" for _ in range(500)]
    cities = ["New York", "San Francisco", "London", "Singapore", "Berlin", "Tokyo", "Toronto", "Sydney"]

    # Assign primary attributes to users
    user_profiles = {}
    for uid in user_ids:
        user_profiles[uid] = {
            "primary_card": random.choice(card_ids),
            "primary_device": random.choice(device_ids),
            "primary_ip": random.choice(ip_addresses),
            "city": random.choice(cities),
            "mean_amount": round(np.random.gamma(shape=3.0, scale=35.0), 2),
            "std_amount": round(np.random.uniform(10, 40), 2),
        }

    # Synthesize Fraud Rings (colluding accounts sharing cards and devices)
    n_rings = 4
    fraud_rings = []
    for r in range(n_rings):
        ring_users = random.sample(user_ids, k=random.randint(4, 7))
        shared_device = f"DEV_RING_{r:02d}"
        shared_card = f"CARD_RING_{r:02d}"
        fraud_rings.append({
            "users": set(ring_users),
            "device": shared_device,
            "card": shared_card
        })

    records = []
    start_time = datetime(2026, 10, 1, 0, 0, 0)
    current_time = start_time

    for tx_idx in range(n_transactions):
        current_time += timedelta(seconds=random.randint(5, 60))
        user_id = random.choice(user_ids)
        profile = user_profiles[user_id]

        is_fraud = 0
        fraud_type = "NONE"

        # Check if in fraud ring
        in_ring = False
        ring_info = None
        for r_data in fraud_rings:
            if user_id in r_data["users"]:
                in_ring = True
                ring_info = r_data
                break

        # Decide fraud
        if in_ring and random.random() < 0.45:
            is_fraud = 1
            fraud_type = "FRAUD_RING_COLLUSION"
            device_id = ring_info["device"]
            card_id = ring_info["card"]
            amount = round(profile["mean_amount"] * np.random.uniform(4.0, 12.0), 2)
            channel = "ONLINE_TRANSFER"
        elif random.random() < fraud_rate:
            is_fraud = 1
            fraud_type = "ACCOUNT_TAKEOVER"
            device_id = random.choice(device_ids)
            card_id = profile["primary_card"]
            amount = round(profile["mean_amount"] * np.random.uniform(5.0, 20.0), 2)
            channel = "WIRE_TRANSFER"
        else:
            is_fraud = 0
            # Normal variance
            device_id = profile["primary_device"] if random.random() > 0.05 else random.choice(device_ids)
            card_id = profile["primary_card"]
            amount = max(2.0, round(np.random.normal(profile["mean_amount"], profile["std_amount"]), 2))
            channel = random.choice(["POS", "ONLINE_COMMERCE", "ATM_WITHDRAWAL", "P2P_TRANSFER"])

        ip_addr = profile["primary_ip"] if is_fraud == 0 and random.random() > 0.08 else random.choice(ip_addresses)
        loc = profile["city"] if is_fraud == 0 and random.random() > 0.04 else random.choice(cities)

        records.append({
            "transaction_id": f"TX_{tx_idx:08d}",
            "timestamp": current_time.isoformat(),
            "user_id": user_id,
            "card_id": card_id,
            "device_id": device_id,
            "ip_address": ip_addr,
            "location": loc,
            "channel": channel,
            "amount": amount,
            "user_mean_amount": profile["mean_amount"],
            "amount_to_mean_ratio": round(amount / (profile["mean_amount"] + 1e-5), 3),
            "is_fraud": is_fraud,
            "fraud_type": fraud_type
        })

    df = pd.DataFrame(records)

    # Compute rolling velocity features
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp_dt").reset_index(drop=True)

    # Group-based transaction velocity (past 1 hr window)
    df["tx_velocity_1h"] = df.groupby("user_id")["timestamp_dt"].diff().dt.total_seconds().fillna(3600)
    df["tx_velocity_1h"] = (3600.0 / (df["tx_velocity_1h"] + 10.0)).round(2)
    df["device_change_flag"] = (df["device_id"] != df.groupby("user_id")["device_id"].shift(1)).astype(int)

    # Clean intermediate columns
    df.drop(columns=["timestamp_dt"], inplace=True)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Dataset] Saved {len(df)} transactions to {output_path}")

    return df

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "transactions_sample.csv")
    df = generate_fraud_dataset(n_transactions=15000, n_users=1000, output_path=out_file)
    print(f"Generated {len(df)} records. Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
