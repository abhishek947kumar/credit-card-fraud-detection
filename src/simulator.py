"""
Real-time Transaction Stream Simulator.
Generates realistic streaming credit card transactions with configurable
fraud injection frequency, distinct attack patterns, and live streaming buffers.
"""
import random
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import CATEGORY_LIST, HIGH_RISK_CATEGORIES


class TransactionStreamSimulator:
    """
    Simulates a live credit card authorization feed for demonstration,
    stress testing, and real-time fraud alert monitoring.
    """

    def __init__(self, buffer_size: int = 150):
        self.buffer_size = buffer_size
        self.history: deque = deque(maxlen=buffer_size)
        self.is_running = False
        self.fraud_rate = 0.05 # 5% for interactive demo visibility
        self.total_processed = 0
        self.total_fraud_detected = 0

    def generate_random_transaction(self, force_fraud: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate a single realistic transaction.
        """
        is_fraud = (
            force_fraud
            if force_fraud is not None
            else (random.random() < self.fraud_rate)
        )
        
        txn_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"
        user_id = f"USR-{random.randint(1001, 2999)}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not is_fraud:
            # Normal legitimate cardholder transaction
            category = random.choice([
                "grocery", "restaurant_dining", "gas_transport",
                "online_retail", "entertainment"
            ])
            amount = round(float(np.random.lognormal(mean=3.6, sigma=0.7)), 2)
            amount = max(3.50, min(amount, 350.0))
            dist_home = round(float(np.random.exponential(scale=6.0)), 2)
            dist_last = round(float(np.random.exponential(scale=4.0)), 2)
            time_since = round(float(np.random.exponential(scale=240.0) + 10.0), 1)
            vel_1h = int(np.random.poisson(lam=0.3))
            vel_24h = vel_1h + int(np.random.poisson(lam=2.0))
            ratio = round(float(np.random.normal(loc=1.0, scale=0.25)), 2)
            ratio = max(0.2, min(ratio, 2.5))
            dev_trust = round(float(np.random.uniform(75.0, 99.0)), 1)
            hour = datetime.now().hour
            foreign = 0 if random.random() > 0.03 else 1
            pin_fail = 0
            v_vec = np.random.normal(0, 0.9, size=8)

        else:
            # Realistic fraud attack vector
            scenario = random.choice(["cloning", "testing", "takeover", "crypto_heist"])
            
            if scenario == "cloning":
                category = random.choice(["luxury_goods", "electronics", "travel_airline"])
                amount = round(random.uniform(450.0, 1850.0), 2)
                dist_home = round(random.uniform(1500.0, 6500.0), 2)
                dist_last = round(random.uniform(1200.0, 5000.0), 2)
                time_since = round(random.uniform(5.0, 40.0), 1) # Impossible speed
                vel_1h = random.randint(1, 3)
                vel_24h = vel_1h + random.randint(2, 5)
                ratio = round(amount / 65.0, 2)
                dev_trust = round(random.uniform(20.0, 50.0), 1)
                hour = datetime.now().hour
                foreign = 1
                pin_fail = random.choice([0, 1])
                v_vec = np.random.normal(-2.2, 1.2, size=8)

            elif scenario == "testing":
                category = random.choice(["online_retail", "entertainment"])
                amount = round(random.choice([random.uniform(1.0, 4.5), random.uniform(250.0, 800.0)]), 2)
                dist_home = round(random.uniform(20.0, 100.0), 2)
                dist_last = round(random.uniform(5.0, 30.0), 2)
                time_since = round(random.uniform(0.1, 3.0), 1) # Rapid bot
                vel_1h = random.randint(6, 15) # Velocity surge
                vel_24h = vel_1h + random.randint(5, 15)
                ratio = round(amount / 40.0, 2)
                dev_trust = round(random.uniform(15.0, 40.0), 1)
                hour = datetime.now().hour
                foreign = random.choice([0, 1])
                pin_fail = random.choice([0, 1, 2])
                v_vec = np.random.normal(1.9, 1.3, size=8)

            elif scenario == "takeover":
                category = "crypto_forex"
                amount = round(random.uniform(1200.0, 4500.0), 2)
                dist_home = round(random.uniform(80.0, 400.0), 2)
                dist_last = round(random.uniform(40.0, 250.0), 2)
                time_since = round(random.uniform(20.0, 120.0), 1)
                vel_1h = random.randint(2, 5)
                vel_24h = vel_1h + random.randint(3, 7)
                ratio = round(amount / 75.0, 2)
                dev_trust = round(random.uniform(5.0, 25.0), 1) # Compromised device
                hour = random.choice([1, 2, 3, 4, 23]) # Late night
                foreign = 1
                pin_fail = random.choice([1, 2])
                v_vec = np.random.normal(2.5, 1.5, size=8)

            else:
                category = "wire_transfer"
                amount = round(random.uniform(2200.0, 5000.0), 2)
                dist_home = round(random.uniform(200.0, 1200.0), 2)
                dist_last = round(random.uniform(100.0, 600.0), 2)
                time_since = round(random.uniform(10.0, 60.0), 1)
                vel_1h = random.randint(3, 7)
                vel_24h = vel_1h + random.randint(4, 9)
                ratio = round(amount / 90.0, 2)
                dev_trust = round(random.uniform(8.0, 30.0), 1)
                hour = datetime.now().hour
                foreign = 1
                pin_fail = random.choice([1, 2, 3])
                v_vec = np.random.normal(2.8, 1.4, size=8)

        txn = {
            "transaction_id": txn_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "amount": amount,
            "category": category,
            "distance_from_home": dist_home,
            "distance_from_last_txn": dist_last,
            "time_since_last_txn": time_since,
            "velocity_1h": vel_1h,
            "velocity_24h": vel_24h,
            "amount_to_user_mean_ratio": ratio,
            "device_trust_score": dev_trust,
            "hour_of_day": hour,
            "day_of_week": datetime.now().weekday(),
            "is_weekend": 1 if datetime.now().weekday() in [5, 6] else 0,
            "is_online": 1 if category in ["online_retail", "crypto_forex", "wire_transfer"] else random.choice([0, 1]),
            "foreign_transaction": foreign,
            "failed_pin_attempts_24h": pin_fail,
            "v1": round(float(v_vec[0]), 4),
            "v2": round(float(v_vec[1]), 4),
            "v3": round(float(v_vec[2]), 4),
            "v4": round(float(v_vec[3]), 4),
            "v5": round(float(v_vec[4]), 4),
            "v6": round(float(v_vec[5]), 4),
            "v7": round(float(v_vec[6]), 4),
            "v8": round(float(v_vec[7]), 4),
            "is_synthetic_fraud": 1 if is_fraud else 0,
        }
        return txn

    def record_transaction(self, scored_txn: Dict[str, Any]):
        """Append scored transaction to streaming buffer."""
        self.history.appendleft(scored_txn)
        self.total_processed += 1
        if scored_txn.get("risk_tier") in ["HIGH", "MEDIUM"]:
            self.total_fraud_detected += 1

    def get_recent_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.history)[:limit]
