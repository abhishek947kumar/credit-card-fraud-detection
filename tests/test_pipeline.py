"""
Automated End-to-End Test Suite for Fraud Detection System.
Tests data generation, feature engineering, model inference,
explainability diagnostics, and FastAPI REST endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.config import CATEGORY_LIST
from src.data_generator import generate_credit_card_data
from src.explainability import FraudExplainer
from src.features import FraudFeatureEngineer
from src.predictor import FraudPredictor


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def predictor():
    return FraudPredictor(active_model_name="xgboost")


def test_data_generator():
    """Verify synthetic dataset generator produces correct schema and fraud label distribution."""
    df = generate_credit_card_data(n_samples=500, fraud_ratio=0.04, random_state=42)
    assert len(df) == 500
    assert "is_fraud" in df.columns
    assert "amount" in df.columns
    assert "category" in df.columns
    assert "distance_from_last_txn" in df.columns
    assert df["is_fraud"].sum() == 20
    assert df["amount"].min() > 0


def test_feature_engineering():
    """Verify feature engineer transforms raw dataframe and single dictionaries into aligned matrices."""
    fe = FraudFeatureEngineer.load("feature_engineer.joblib")
    df = generate_credit_card_data(n_samples=50, fraud_ratio=0.1)
    X = fe.transform(df)
    
    assert X.shape[0] == 50
    assert X.shape[1] == len(fe.get_input_feature_columns())

    single_txn = {
        "amount": 75.0,
        "category": "grocery",
        "distance_from_home": 3.0,
        "distance_from_last_txn": 1.2,
        "time_since_last_txn": 150.0,
        "velocity_1h": 0,
        "velocity_24h": 2,
        "amount_to_user_mean_ratio": 1.0,
        "device_trust_score": 95.0,
        "hour_of_day": 12,
        "day_of_week": 1,
        "is_weekend": 0,
        "is_online": 0,
        "foreign_transaction": 0,
        "failed_pin_attempts_24h": 0,
    }
    X_single = fe.transform_single(single_txn)
    assert X_single.shape == (1, len(fe.get_input_feature_columns()))


def test_predictor_scoring(predictor):
    """Verify predictor scores normal and fraudulent transactions accurately with proper actions."""
    # 1. Normal transaction
    normal_txn = {
        "transaction_id": "TXN-TEST-NORM",
        "amount": 18.50,
        "category": "grocery",
        "distance_from_last_txn": 2.0,
        "time_since_last_txn": 300.0,
        "velocity_1h": 0,
        "velocity_24h": 2,
        "amount_to_user_mean_ratio": 0.9,
        "device_trust_score": 96.0,
        "foreign_transaction": 0,
        "failed_pin_attempts_24h": 0,
    }
    res_norm = predictor.predict_single(normal_txn)
    assert res_norm["risk_tier"] == "LOW"
    assert res_norm["action"] == "AUTO_APPROVE"
    assert res_norm["risk_score"] < 25.0

    # 2. Fraudulent attack: Card Cloning Teleportation
    fraud_txn = {
        "transaction_id": "TXN-TEST-FRAUD",
        "amount": 2800.0,
        "category": "luxury_goods",
        "distance_from_last_txn": 4500.0, # 4500 km in 20 min
        "time_since_last_txn": 20.0,
        "velocity_1h": 3,
        "velocity_24h": 6,
        "amount_to_user_mean_ratio": 15.0,
        "device_trust_score": 15.0,
        "foreign_transaction": 1,
        "failed_pin_attempts_24h": 2,
    }
    res_fraud = predictor.predict_single(fraud_txn)
    assert res_fraud["risk_tier"] == "HIGH"
    assert res_fraud["action"] == "DECLINE_AND_FREEZE"
    assert res_fraud["risk_score"] > 70.0
    assert len(res_fraud["risk_factors"]) > 0


def test_explainability():
    """Verify FraudExplainer catches specific fraud drivers."""
    explainer = FraudExplainer()
    txn = {
        "amount": 3500.0,
        "category": "crypto_forex",
        "distance_from_last_txn": 4200.0,
        "time_since_last_txn": 15.0, # >16,000 km/h
        "velocity_1h": 9,
        "device_trust_score": 10.0,
        "failed_pin_attempts_24h": 3,
        "foreign_transaction": 1,
        "amount_to_user_mean_ratio": 12.0,
    }
    factors = explainer.explain_transaction(txn, prob=0.98)
    factor_names = [f["factor"] for f in factors]
    
    assert any("Impossible Travel" in name for name in factor_names)
    assert any("Velocity Surge" in name for name in factor_names)
    assert any("Failed PIN" in name for name in factor_names)
    assert any("Untrusted" in name for name in factor_names)


def test_api_endpoints(client):
    """Test FastAPI health, predict, models metadata, and simulator generation."""
    # Health
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # Single Predict
    pred_payload = {
        "amount": 25.00,
        "category": "restaurant_dining",
        "distance_from_home": 4.0,
        "distance_from_last_txn": 1.5,
        "time_since_last_txn": 180.0,
        "velocity_1h": 0,
        "velocity_24h": 2,
        "amount_to_user_mean_ratio": 1.0,
        "device_trust_score": 90.0,
        "hour_of_day": 13,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_online": 0,
        "foreign_transaction": 0,
        "failed_pin_attempts_24h": 0,
    }
    r = client.post("/api/predict", json=pred_payload)
    assert r.status_code == 200
    body = r.json()
    assert "risk_score" in body
    assert "action" in body
    assert "badge_color" in body

    # Models metadata
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "xgboost" in data["models"]

    # Simulator generate
    r = client.post("/api/simulator/generate", json={"count": 2})
    assert r.status_code == 200
    assert len(r.json()["transactions"]) == 2
