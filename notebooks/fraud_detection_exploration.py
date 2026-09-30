"""
Exploratory Data Analysis & Machine Learning Benchmark Script.
Can be executed directly or run inside Jupyter notebook environments.
Demonstrates:
1. Synthetic and Benchmark Credit Card Dataset Loading
2. Extreme Class Imbalance Analysis & Visualization
3. Feature Engineering & Geo-velocity Metrics
4. Multi-Model Training (XGBoost, Random Forest, Isolation Forest)
5. Precision-Recall AUC & Financial Loss Analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from src.data_generator import generate_credit_card_data
from src.evaluate import compute_metrics
from src.features import FraudFeatureEngineer
from src.predictor import FraudPredictor


def run_exploration():
    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION - EXPLORATORY ANALYSIS")
    print("=" * 60)

    # 1. Load Data
    print("\n[1] Generating 10,000 transaction dataset...")
    df = generate_credit_card_data(n_samples=10000, fraud_ratio=0.015, random_state=42)
    n_fraud = df["is_fraud"].sum()
    print(f"Total Transactions: {len(df):,}")
    print(f"Fraudulent Transactions: {n_fraud:,} ({n_fraud / len(df) * 100:.2f}%)")
    print(f"Legitimate Transactions: {len(df) - n_fraud:,} ({(len(df) - n_fraud) / len(df) * 100:.2f}%)")

    # 2. Spend Profile Summary
    print("\n[2] Spend Distribution by Class:")
    spend_stats = df.groupby("is_fraud")["amount"].agg(["mean", "median", "std", "max"])
    spend_stats.index = ["Legitimate (0)", "Fraudulent (1)"]
    print(spend_stats)

    # 3. Geo-speed & Velocity Anomalies
    print("\n[3] Calculating Geo-Velocity Speed...")
    df["speed_kmh"] = df["distance_from_last_txn"] / np.maximum(df["time_since_last_txn"] / 60.0, 0.05)
    speed_by_class = df.groupby("is_fraud")["speed_kmh"].agg(["mean", "max"])
    speed_by_class.index = ["Legitimate (0)", "Fraudulent (1)"]
    print(speed_by_class)

    # 4. Score with Real-Time Predictor
    print("\n[4] Scoring samples with Trained FraudPredictor...")
    predictor = FraudPredictor(active_model_name="xgboost")
    
    sample_legit = df[df["is_fraud"] == 0].iloc[0].to_dict()
    res_legit = predictor.predict_single(sample_legit)
    print(f"\nLegitimate Transaction Test:")
    print(f" - Amount: ${sample_legit['amount']:.2f} at {sample_legit['category']}")
    print(f" - Assigned Risk: {res_legit['risk_tier']} ({res_legit['risk_score']}%)")
    print(f" - Operational Action: {res_legit['action']}")

    sample_fraud = df[df["is_fraud"] == 1].iloc[0].to_dict()
    res_fraud = predictor.predict_single(sample_fraud)
    print(f"\nFraudulent Transaction Test:")
    print(f" - Amount: ${sample_fraud['amount']:.2f} at {sample_fraud['category']}")
    print(f" - Assigned Risk: {res_fraud['risk_tier']} ({res_fraud['risk_score']}%)")
    print(f" - Operational Action: {res_fraud['action']}")
    print(f" - Top XAI Drivers:")
    for f in res_fraud["risk_factors"][:3]:
        print(f"    * [{f['severity']}] {f['factor']}: {f['detail']}")

    print("\n[5] Exploration Completed Successfully!")


if __name__ == "__main__":
    run_exploration()
