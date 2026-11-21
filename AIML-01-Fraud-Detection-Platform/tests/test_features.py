"""
Unit tests for Feature Store materialization and online retrieval.
"""

import pytest
import pandas as pd
from features.feature_store import FraudFeatureStore

def test_feature_store_materialization(tmp_path):
    db_file = str(tmp_path / "test_store.db")
    store = FraudFeatureStore(db_path=db_file)
    
    sample_df = pd.DataFrame([
        {"user_id": "U100", "amount": 50.0, "device_id": "D1", "location": "NYC", "timestamp": "2026-10-01T10:00:00"},
        {"user_id": "U100", "amount": 150.0, "device_id": "D1", "location": "NYC", "timestamp": "2026-10-01T11:00:00"},
    ])
    store.materialize_from_dataframe(sample_df)
    
    feats = store.get_online_features("U100", "D2", 500.0, "London")
    assert feats["user_id"] == "U100"
    assert feats["device_change_flag"] == 1
    assert feats["location_change_flag"] == 1
    assert feats["amount_to_mean_ratio"] == 5.0
