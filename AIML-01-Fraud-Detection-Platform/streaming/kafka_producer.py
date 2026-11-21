"""
Kafka Transaction Producer & Stream Simulator.
Replays transaction streams in real-time, delivering batches or streaming events
into Apache Kafka or passing directly to the scoring service in simulation mode.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data.dataset_generator import generate_fraud_dataset

# Try importing KafkaProducer
try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

class TransactionStreamProducer:
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "financial-transactions"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None

        if KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    request_timeout_ms=5000
                )
                print(f"[Kafka] Connected to broker at {self.bootstrap_servers}, topic: {self.topic}")
            except Exception as e:
                print(f"[Kafka Warning] Could not connect to Kafka broker ({e}). Falling back to stream simulator.")
                self.producer = None
        else:
            print("[Kafka Notice] kafka-python not installed. Running in high-fidelity simulated streaming mode.")

    def emit_transaction(self, record: dict):
        if self.producer:
            self.producer.send(self.topic, value=record)
            self.producer.flush()
        return record

    def run_replay(self, csv_path: str = None, interval_seconds: float = 0.5, max_events: int = 50):
        print(f"[Stream Replay] Starting replay (interval: {interval_seconds}s, max_events: {max_events})...")
        if csv_path and os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            records = df.to_dict(orient="records")
        else:
            df = generate_fraud_dataset(n_transactions=max_events)
            records = df.to_dict(orient="records")

        count = 0
        for tx in records[:max_events]:
            payload = {
                "transaction_id": tx["transaction_id"],
                "user_id": tx["user_id"],
                "card_id": tx["card_id"],
                "device_id": tx["device_id"],
                "amount": float(tx["amount"]),
                "channel": tx["channel"],
                "location": tx["location"],
                "ip_address": tx["ip_address"],
                "timestamp": datetime.utcnow().isoformat()
            }
            self.emit_transaction(payload)
            count += 1
            print(f"[{count}/{max_events}] Emitted {payload['transaction_id']} | User: {payload['user_id']} | Amount: ${payload['amount']} | Channel: {payload['channel']}")
            time.sleep(interval_seconds)

        print("[Stream Replay] Replay stream completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between emitted events")
    parser.add_argument("--events", type=int, default=20, help="Number of events to emit")
    args = parser.parse_args()

    streamer = TransactionStreamProducer()
    streamer.run_replay(interval_seconds=args.interval, max_events=args.events)
