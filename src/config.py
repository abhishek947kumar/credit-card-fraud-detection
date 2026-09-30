"""
Configuration and constants for the Credit Card Fraud Detection System.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Risk Thresholds (Probabilities 0.0 to 1.0)
RISK_THRESHOLD_LOW = 0.25      # Below this: Auto-Approve (Normal)
RISK_THRESHOLD_MEDIUM = 0.65   # Between Low and Medium: Flag for Step-up Auth / Review
                               # Above Medium: Hard Block / Decline

# Financial Cost Matrix (USD) for Business ROI Evaluation
COST_FALSE_NEGATIVE = 385.0    # Undetected fraud (chargeback, goods lost, fee)
COST_FALSE_POSITIVE = 12.50    # Customer friction, review queue cost
AVERAGE_INTERCHANGE_FEE = 0.02 # 2% interchange revenue preserved

# Features Schema
NUMERICAL_FEATURES = [
    "amount",
    "distance_from_home",
    "distance_from_last_txn",
    "time_since_last_txn",
    "velocity_1h",
    "velocity_24h",
    "amount_to_user_mean_ratio",
    "device_trust_score",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_online",
    "foreign_transaction",
    "failed_pin_attempts_24h",
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
]

CATEGORICAL_FEATURES = [
    "category"
]

CATEGORY_LIST = [
    "grocery",
    "online_retail",
    "electronics",
    "travel_airline",
    "restaurant_dining",
    "crypto_forex",
    "wire_transfer",
    "entertainment",
    "gas_transport",
    "luxury_goods"
]

HIGH_RISK_CATEGORIES = {"crypto_forex", "wire_transfer", "electronics", "luxury_goods"}

TARGET_COLUMN = "is_fraud"

# Default Model Parameters
RANDOM_STATE = 42
DEFAULT_FRAUD_RATIO = 0.012  # ~1.2% fraud in generated synthetic data
