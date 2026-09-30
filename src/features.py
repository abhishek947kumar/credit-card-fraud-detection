"""
Feature Engineering and Data Preprocessing Pipeline.
Implements domain-specific fraud risk indicators:
- Geo-velocity speed anomaly (km/h between successive transactions)
- Log monetary transformations
- Risk category indicators
- Resampling techniques (SMOTE, Random Under-Sampling)
- Scaler fitting and serialization
"""
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler

from src.config import (
    CATEGORY_LIST,
    HIGH_RISK_CATEGORIES,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


class FraudFeatureEngineer:
    """
    Handles feature engineering, scaling, and categorical one-hot alignment
    for both batch datasets and single real-time transaction payloads.
    """

    def __init__(self, scaler_path: Optional[str] = None):
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.categories = sorted(CATEGORY_LIST)
        
        if scaler_path and (MODELS_DIR / scaler_path).exists():
            self.load(scaler_path)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive advanced fraud features from raw transaction fields.
        """
        data = df.copy()

        # Fill default values for missing columns if necessary
        default_cols = {
            "amount": 50.0,
            "category": "grocery",
            "distance_from_home": 5.0,
            "distance_from_last_txn": 2.0,
            "time_since_last_txn": 120.0,
            "velocity_1h": 0,
            "velocity_24h": 1,
            "amount_to_user_mean_ratio": 1.0,
            "device_trust_score": 85.0,
            "hour_of_day": 14,
            "day_of_week": 2,
            "is_weekend": 0,
            "is_online": 1,
            "foreign_transaction": 0,
            "failed_pin_attempts_24h": 0,
            "v1": 0.0, "v2": 0.0, "v3": 0.0, "v4": 0.0,
            "v5": 0.0, "v6": 0.0, "v7": 0.0, "v8": 0.0,
        }
        for col, val in default_cols.items():
            if col not in data.columns:
                data[col] = val

        # 1. Log-transformed amount for monetary normalization
        data["amount_log"] = np.log1p(np.maximum(data["amount"], 0))

        # 2. Geo-speed: km/h between current and last transaction
        # Speed > 800 km/h is physical teleportation (card cloning vector)
        time_hours = np.maximum(data["time_since_last_txn"] / 60.0, 0.05)
        data["geo_speed_kmh"] = data["distance_from_last_txn"] / time_hours
        data["is_impossible_travel"] = (data["geo_speed_kmh"] > 750.0).astype(int)

        # 3. Velocity acceleration: 1h velocity surge vs 24h baseline
        hourly_baseline = np.maximum(data["velocity_24h"] / 24.0, 0.1)
        data["velocity_acceleration"] = data["velocity_1h"] / hourly_baseline

        # 4. Late-night fraud window (1 AM to 5 AM)
        data["is_night_txn"] = data["hour_of_day"].isin([1, 2, 3, 4, 5]).astype(int)

        # 5. High-risk merchant category flag
        data["is_high_risk_category"] = data["category"].apply(
            lambda c: 1 if c in HIGH_RISK_CATEGORIES else 0
        )

        # 6. Device suspicion index
        # Low trust score (<40) + online or foreign
        data["device_suspicion_score"] = (
            (100.0 - data["device_trust_score"]) / 100.0
            + data["foreign_transaction"] * 0.5
            + data["failed_pin_attempts_24h"] * 0.8
        )

        # 7. Category one-hot encoding (fixed schema to guarantee alignment)
        for cat in self.categories:
            col_name = f"cat_{cat}"
            data[col_name] = (data["category"] == cat).astype(int)

        return data

    def get_input_feature_columns(self) -> List[str]:
        """
        Return the final list of engineered feature names used for modeling.
        """
        base_features = [
            "amount",
            "amount_log",
            "distance_from_home",
            "distance_from_last_txn",
            "time_since_last_txn",
            "geo_speed_kmh",
            "is_impossible_travel",
            "velocity_1h",
            "velocity_24h",
            "velocity_acceleration",
            "amount_to_user_mean_ratio",
            "device_trust_score",
            "device_suspicion_score",
            "hour_of_day",
            "is_night_txn",
            "day_of_week",
            "is_weekend",
            "is_online",
            "foreign_transaction",
            "failed_pin_attempts_24h",
            "is_high_risk_category",
            "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
        ]
        cat_features = [f"cat_{cat}" for cat in self.categories]
        return base_features + cat_features

    def fit(self, df: pd.DataFrame) -> "FraudFeatureEngineer":
        """
        Fit scaler on training data.
        """
        engineered_df = self.engineer_features(df)
        self.feature_names = self.get_input_feature_columns()
        X = engineered_df[self.feature_names].values
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform raw transactions into scaled feature matrix.
        """
        if not self.is_fitted:
            raise ValueError("FraudFeatureEngineer must be fitted before transforming data.")
        engineered_df = self.engineer_features(df)
        self.feature_names = self.get_input_feature_columns()
        X = engineered_df[self.feature_names].values
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_single(self, txn_dict: Dict) -> np.ndarray:
        """
        Convert a single transaction dictionary into a 2D numpy row ready for inference.
        """
        df = pd.DataFrame([txn_dict])
        return self.transform(df)

    def save(self, filename: str = "feature_engineer.joblib"):
        """Save fitted feature engineer and scaler to disk."""
        path = MODELS_DIR / filename
        joblib.dump(self, path)
        print(f"Feature engineer saved to {path}")

    @classmethod
    def load(cls, filename: str = "feature_engineer.joblib") -> "FraudFeatureEngineer":
        """Load fitted feature engineer from disk."""
        path = MODELS_DIR / filename
        return joblib.load(path)


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote",
    sampling_strategy: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample training data to address extreme class imbalance.
    
    Args:
        X: Feature matrix
        y: Labels
        strategy: 'smote', 'undersample', or 'none'
        sampling_strategy: Desired ratio of minority to majority class (e.g. 0.2 = 1:5 ratio)
        random_state: Seed
    """
    if strategy == "smote":
        print(f"Applying SMOTE resampling (target minority ratio: {sampling_strategy})...")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        return smote.fit_resample(X, y)
    elif strategy == "undersample":
        print(f"Applying Random Under-Sampling (target minority ratio: {sampling_strategy})...")
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        return rus.fit_resample(X, y)
    else:
        return X, y
