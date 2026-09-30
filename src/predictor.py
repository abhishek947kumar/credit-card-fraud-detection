"""
Inference & Fraud Risk Scoring Engine.
Loads trained models, performs feature transformation, executes model inference,
evaluates risk tiers (Low/Medium/High), and attaches explainability diagnostics.
"""
import time
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_MEDIUM,
)
from src.explainability import FraudExplainer
from src.features import FraudFeatureEngineer
# Import BlendedFraudEnsemble so joblib can unpickle ensemble models
import __main__
from src.train import BlendedFraudEnsemble
setattr(__main__, "BlendedFraudEnsemble", BlendedFraudEnsemble)


class FraudPredictor:
    """
    Production-grade predictor that manages model switching, feature engineering,
    latency tracking, risk tier classification, and explainability.
    """

    MODEL_FILES = {
        "xgboost": "xgboost_model.joblib",
        "random_forest": "random_forest_model.joblib",
        "hist_gradient_boost": "hist_gradient_boost_model.joblib",
        "logistic_regression": "logistic_regression_model.joblib",
        "isolation_forest": "isolation_forest_model.joblib",
        "ensemble": "ensemble_model.joblib",
    }

    def __init__(self, active_model_name: str = "xgboost"):
        self.active_model_name = active_model_name
        self.models: Dict[str, Any] = {}
        self.feature_engineer = FraudFeatureEngineer.load("feature_engineer.joblib")
        self.explainer = FraudExplainer()
        self.load_model(active_model_name)

    def load_model(self, model_name: str):
        """Load a specific model from disk cache or memory."""
        if model_name not in self.MODEL_FILES:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_FILES.keys())}")
        
        if model_name not in self.models:
            model_path = MODELS_DIR / self.MODEL_FILES[model_name]
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.models[model_name] = joblib.load(model_path)
            
        self.active_model_name = model_name

    def set_active_model(self, model_name: str):
        self.load_model(model_name)

    def predict_single(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single real-time transaction payload.
        """
        t0 = time.perf_counter()
        
        # Ensure default values for any optional fields
        clean_txn = self._normalize_txn_input(txn_dict)
        
        # Transform features
        df_row = pd.DataFrame([clean_txn])
        X = self.feature_engineer.transform(df_row)
        
        # Execute inference
        model = self.models[self.active_model_name]
        
        if self.active_model_name == "isolation_forest":
            raw_score = model.decision_function(X)[0]
            # Map anomaly score to probability
            prob = float(1.0 / (1.0 + np.exp(raw_score * 6.0)))
        else:
            prob = float(model.predict_proba(X)[0, 1])

        latency_ms = (time.perf_counter() - t0) * 1000.0
        risk_score = round(prob * 100.0, 1)

        # Risk Tier & Action Recommendation
        if prob < RISK_THRESHOLD_LOW:
            risk_tier = "LOW"
            action = "AUTO_APPROVE"
            action_desc = "Transaction authorized instantly. Zero customer friction."
            color = "emerald"
        elif prob < RISK_THRESHOLD_MEDIUM:
            risk_tier = "MEDIUM"
            action = "STEP_UP_AUTH"
            action_desc = "Friction triggered: Dispatch SMS OTP / 3D Secure / Analyst Review."
            color = "amber"
        else:
            risk_tier = "HIGH"
            action = "DECLINE_AND_FREEZE"
            action_desc = "Critical threat detected. Transaction declined and card locked."
            color = "rose"

        # Generate Explainable Risk Drivers
        risk_factors = self.explainer.explain_transaction(clean_txn, prob)

        return {
            "transaction_id": clean_txn.get("transaction_id", "TXN-UNKNOWN"),
            "timestamp": clean_txn.get("timestamp", ""),
            "user_id": clean_txn.get("user_id", "USR-UNKNOWN"),
            "amount": clean_txn.get("amount", 0.0),
            "category": clean_txn.get("category", "general"),
            "fraud_probability": round(prob, 4),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "action": action,
            "action_description": action_desc,
            "badge_color": color,
            "active_model": self.active_model_name,
            "latency_ms": round(latency_ms, 2),
            "risk_factors": risk_factors,
            "details": clean_txn,
        }

    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch score a list of transactions efficiently.
        """
        results = [self.predict_single(txn) for txn in transactions]
        return results

    def _normalize_txn_input(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Fill defaults for missing fields to avoid runtime failures."""
        normalized = dict(txn)
        defaults = {
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
        for k, v in defaults.items():
            if k not in normalized or normalized[k] is None:
                normalized[k] = v
        return normalized
