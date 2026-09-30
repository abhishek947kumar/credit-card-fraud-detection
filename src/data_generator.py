"""
Synthetic Credit Card Transaction Dataset Generator.
Simulates realistic financial transactions, user spending baselines,
and distinct fraud attack vectors (Card Cloning, Account Takeover,
Card Testing Velocity Spikes, High-Value Midnight Drains).
"""
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

from src.config import (
    CATEGORY_LIST,
    DEFAULT_FRAUD_RATIO,
    RANDOM_STATE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


def generate_credit_card_data(
    n_samples: int = 15000,
    fraud_ratio: float = DEFAULT_FRAUD_RATIO,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate synthetic realistic credit card transactions dataset.
    
    Args:
        n_samples: Total number of transactions to generate
        fraud_ratio: Target proportion of fraudulent transactions (~0.01 - 0.02)
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame containing engineered transaction features and 'is_fraud' label
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # 1. Base normal user behavior (Log-normal amount, reasonable velocities)
    # Amounts: lognormal centered around $45-$80
    normal_amounts = np.random.lognormal(mean=3.7, sigma=0.85, size=n_normal)
    normal_amounts = np.clip(normal_amounts, 2.50, 850.00)
    
    # Normal distance from home (0 to 35 km)
    normal_dist_home = np.random.exponential(scale=8.0, size=n_normal)
    normal_dist_home = np.clip(normal_dist_home, 0.1, 80.0)
    
    # Distance from last txn (mostly small, occasional trips)
    normal_dist_last = np.random.exponential(scale=5.0, size=n_normal)
    
    # Time since last transaction (hours converted to minutes)
    normal_time_since = np.random.exponential(scale=360.0, size=n_normal) + 5.0
    
    # Velocity (1h and 24h)
    normal_vel_1h = np.random.poisson(lam=0.4, size=n_normal)
    normal_vel_24h = normal_vel_1h + np.random.poisson(lam=2.2, size=n_normal)
    
    # Ratio to user historical mean
    normal_ratio = np.random.normal(loc=1.0, scale=0.35, size=n_normal)
    normal_ratio = np.clip(normal_ratio, 0.1, 3.0)
    
    # Device trust score (0-100, high for normal users)
    normal_device_trust = np.random.beta(a=8, b=2, size=n_normal) * 100.0
    
    # Time features (Normal distribution around daytime/evening, peak 11am-8pm)
    normal_hours = (np.random.normal(loc=14.5, scale=4.0, size=n_normal).astype(int)) % 24
    normal_dow = np.random.randint(0, 7, size=n_normal)
    normal_weekend = np.isin(normal_dow, [5, 6]).astype(int)
    
    # Flags
    normal_online = np.random.binomial(n=1, p=0.45, size=n_normal)
    normal_foreign = np.random.binomial(n=1, p=0.03, size=n_normal)
    normal_failed_pin = np.random.choice([0, 1], size=n_normal, p=[0.985, 0.015])
    
    # Categories (skewed towards routine spending)
    cat_probs_normal = [0.28, 0.20, 0.08, 0.05, 0.18, 0.01, 0.01, 0.08, 0.10, 0.01]
    normal_categories = np.random.choice(CATEGORY_LIST, size=n_normal, p=cat_probs_normal)
    
    # Synthetic PCA features V1..V8 (Gaussian with low variance)
    normal_v = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, 8))

    # 2. Fraud patterns simulation (4 real-world attack vectors)
    # Vector A: High-value midnight theft / account takeover (35% of frauds)
    # Vector B: Card cloning geo-teleportation (25% of frauds)
    # Vector C: Card testing micro-auth velocity storm (25% of frauds)
    # Vector D: High-risk merchant / foreign brute-force (15% of frauds)
    
    fraud_amounts = []
    fraud_dist_home = []
    fraud_dist_last = []
    fraud_time_since = []
    fraud_vel_1h = []
    fraud_vel_24h = []
    fraud_ratio = []
    fraud_device_trust = []
    fraud_hours = []
    fraud_dow = []
    fraud_weekend = []
    fraud_online = []
    fraud_foreign = []
    fraud_failed_pin = []
    fraud_categories = []
    fraud_v = []
    
    for _ in range(n_fraud):
        attack_type = np.random.choice(["takeover", "cloning", "testing", "foreign_brute"])
        
        if attack_type == "takeover":
            # High amount, late night, crypto or electronics, low device trust
            amt = float(np.random.uniform(900.0, 4800.0))
            d_home = float(np.random.uniform(30.0, 300.0))
            d_last = float(np.random.uniform(20.0, 500.0))
            t_last = float(np.random.uniform(10.0, 180.0))
            v_1h = int(np.random.randint(1, 4))
            v_24h = int(v_1h + np.random.randint(2, 6))
            ratio = float(amt / np.random.uniform(50.0, 120.0))
            dev_trust = float(np.random.uniform(5.0, 35.0))
            hr = int(np.random.choice([1, 2, 3, 4, 23]))
            dow = int(np.random.randint(0, 7))
            wknd = int(dow in [5, 6])
            online = 1
            foreign = int(np.random.choice([0, 1], p=[0.4, 0.6]))
            pin_fail = int(np.random.choice([0, 1], p=[0.7, 0.3]))
            cat = str(np.random.choice(["crypto_forex", "wire_transfer", "electronics", "luxury_goods"]))
            v_vec = np.random.normal(loc=2.2, scale=1.3, size=8)

        elif attack_type == "cloning":
            # Geo teleportation: huge distance in tiny time span
            amt = float(np.random.uniform(250.0, 2200.0))
            d_home = float(np.random.uniform(1200.0, 7500.0))
            d_last = float(np.random.uniform(1000.0, 6000.0))
            t_last = float(np.random.uniform(5.0, 45.0)) # Impossible speed!
            v_1h = int(np.random.randint(1, 3))
            v_24h = int(v_1h + np.random.randint(1, 4))
            ratio = float(amt / np.random.uniform(40.0, 90.0))
            dev_trust = float(np.random.uniform(20.0, 60.0))
            hr = int(np.random.randint(0, 24))
            dow = int(np.random.randint(0, 7))
            wknd = int(dow in [5, 6])
            online = 0 # POS clone
            foreign = 1
            pin_fail = int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
            cat = str(np.random.choice(["luxury_goods", "electronics", "travel_airline"]))
            v_vec = np.random.normal(loc=-2.0, scale=1.4, size=8)

        elif attack_type == "testing":
            # Card bot testing: rapid fire small or escalating transactions
            amt = float(np.random.choice([np.random.uniform(1.0, 5.0), np.random.uniform(350.0, 1200.0)]))
            d_home = float(np.random.uniform(10.0, 200.0))
            d_last = float(np.random.uniform(5.0, 50.0))
            t_last = float(np.random.uniform(0.1, 4.0)) # Rapid automated bot
            v_1h = int(np.random.randint(5, 16)) # High velocity
            v_24h = int(v_1h + np.random.randint(4, 18))
            ratio = float(amt / 60.0)
            dev_trust = float(np.random.uniform(10.0, 45.0))
            hr = int(np.random.randint(0, 24))
            dow = int(np.random.randint(0, 7))
            wknd = int(dow in [5, 6])
            online = 1
            foreign = int(np.random.choice([0, 1], p=[0.5, 0.5]))
            pin_fail = int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
            cat = str(np.random.choice(["online_retail", "entertainment", "crypto_forex"]))
            v_vec = np.random.normal(loc=1.8, scale=1.5, size=8)

        else: # foreign_brute
            amt = float(np.random.uniform(400.0, 3100.0))
            d_home = float(np.random.uniform(800.0, 5000.0))
            d_last = float(np.random.uniform(500.0, 3000.0))
            t_last = float(np.random.uniform(15.0, 240.0))
            v_1h = int(np.random.randint(2, 6))
            v_24h = int(v_1h + np.random.randint(3, 8))
            ratio = float(amt / 55.0)
            dev_trust = float(np.random.uniform(0.0, 25.0))
            hr = int(np.random.randint(0, 24))
            dow = int(np.random.randint(0, 7))
            wknd = int(dow in [5, 6])
            online = int(np.random.choice([0, 1], p=[0.3, 0.7]))
            foreign = 1
            pin_fail = int(np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
            cat = str(np.random.choice(["wire_transfer", "crypto_forex", "electronics"]))
            v_vec = np.random.normal(loc=2.5, scale=1.6, size=8)

        fraud_amounts.append(round(amt, 2))
        fraud_dist_home.append(round(d_home, 2))
        fraud_dist_last.append(round(d_last, 2))
        fraud_time_since.append(round(t_last, 2))
        fraud_vel_1h.append(v_1h)
        fraud_vel_24h.append(v_24h)
        fraud_ratio.append(round(ratio, 2))
        fraud_device_trust.append(round(dev_trust, 2))
        fraud_hours.append(hr)
        fraud_dow.append(dow)
        fraud_weekend.append(wknd)
        fraud_online.append(online)
        fraud_foreign.append(foreign)
        fraud_failed_pin.append(pin_fail)
        fraud_categories.append(cat)
        fraud_v.append(v_vec)

    fraud_v = np.array(fraud_v)

    # Combine normal and fraud
    amounts = np.concatenate([np.round(normal_amounts, 2), np.array(fraud_amounts)])
    dist_home = np.concatenate([np.round(normal_dist_home, 2), np.array(fraud_dist_home)])
    dist_last = np.concatenate([np.round(normal_dist_last, 2), np.array(fraud_dist_last)])
    time_since = np.concatenate([np.round(normal_time_since, 2), np.array(fraud_time_since)])
    vel_1h = np.concatenate([normal_vel_1h, np.array(fraud_vel_1h)])
    vel_24h = np.concatenate([normal_vel_24h, np.array(fraud_vel_24h)])
    ratios = np.concatenate([np.round(normal_ratio, 2), np.array(fraud_ratio)])
    dev_trusts = np.concatenate([np.round(normal_device_trust, 2), np.array(fraud_device_trust)])
    hours = np.concatenate([normal_hours, np.array(fraud_hours)])
    dows = np.concatenate([normal_dow, np.array(fraud_dow)])
    weekends = np.concatenate([normal_weekend, np.array(fraud_weekend)])
    onlines = np.concatenate([normal_online, np.array(fraud_online)])
    foreigns = np.concatenate([normal_foreign, np.array(fraud_foreign)])
    failed_pins = np.concatenate([normal_failed_pin, np.array(fraud_failed_pin)])
    categories = np.concatenate([normal_categories, np.array(fraud_categories)])
    v_matrix = np.vstack([normal_v, fraud_v])
    
    labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])
    
    # Generate timestamps and IDs
    base_time = datetime.now() - timedelta(days=14)
    timestamps = [
        (base_time + timedelta(minutes=float(i * 1.35) + np.random.uniform(0, 1.2))).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n_samples)
    ]
    txn_ids = [f"TXN-{uuid.uuid4().hex[:10].upper()}" for _ in range(n_samples)]
    user_ids = [f"USR-{np.random.randint(1000, 2500):04d}" for _ in range(n_samples)]

    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "timestamp": timestamps,
        "user_id": user_ids,
        "amount": amounts,
        "category": categories,
        "distance_from_home": dist_home,
        "distance_from_last_txn": dist_last,
        "time_since_last_txn": time_since,
        "velocity_1h": vel_1h,
        "velocity_24h": vel_24h,
        "amount_to_user_mean_ratio": ratios,
        "device_trust_score": dev_trusts,
        "hour_of_day": hours,
        "day_of_week": dows,
        "is_weekend": weekends,
        "is_online": onlines,
        "foreign_transaction": foreigns,
        "failed_pin_attempts_24h": failed_pins,
        "v1": np.round(v_matrix[:, 0], 4),
        "v2": np.round(v_matrix[:, 1], 4),
        "v3": np.round(v_matrix[:, 2], 4),
        "v4": np.round(v_matrix[:, 3], 4),
        "v5": np.round(v_matrix[:, 4], 4),
        "v6": np.round(v_matrix[:, 5], 4),
        "v7": np.round(v_matrix[:, 6], 4),
        "v8": np.round(v_matrix[:, 7], 4),
        "is_fraud": labels
    })

    # Shuffle rows to avoid order bias
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def generate_and_save_datasets(
    train_samples: int = 20000,
    test_samples: int = 5000,
    fraud_ratio: float = DEFAULT_FRAUD_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate train and test sets and persist to CSV.
    """
    print(f"Generating synthetic credit card dataset: {train_samples} train, {test_samples} test...")
    train_df = generate_credit_card_data(n_samples=train_samples, fraud_ratio=fraud_ratio, random_state=42)
    test_df = generate_credit_card_data(n_samples=test_samples, fraud_ratio=fraud_ratio, random_state=99)
    
    train_path = RAW_DATA_DIR / "transactions_train.csv"
    test_path = RAW_DATA_DIR / "transactions_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved training dataset: {train_path} ({len(train_df)} rows, {train_df['is_fraud'].sum()} frauds)")
    print(f"Saved testing dataset: {test_path} ({len(test_df)} rows, {test_df['is_fraud'].sum()} frauds)")
    return train_df, test_df


if __name__ == "__main__":
    generate_and_save_datasets()
