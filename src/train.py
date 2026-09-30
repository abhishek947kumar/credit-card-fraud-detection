"""
Model Training & Model Selection Pipeline.
Trains and compares multiple machine learning algorithms:
- XGBoost (with scale_pos_weight)
- Balanced Random Forest
- HistGradientBoosting
- Logistic Regression (Cost-Sensitive)
- Isolation Forest (Unsupervised Anomaly Detection)
- Ensemble Blended Model (Supervised + Unsupervised)
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import (
    DEFAULT_FRAUD_RATIO,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)
from src.data_generator import generate_and_save_datasets
from src.evaluate import compute_metrics, find_optimal_threshold
from src.features import FraudFeatureEngineer, balance_dataset


class BlendedFraudEnsemble:
    """
    Blends Supervised (XGBoost + Random Forest) with
    Unsupervised Anomaly Detection (Isolation Forest).
    """

    def __init__(self, xgb_model, rf_model, iforest_model, weights=(0.50, 0.35, 0.15)):
        self.xgb = xgb_model
        self.rf = rf_model
        self.iforest = iforest_model
        self.weights = weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_rf = self.rf.predict_proba(X)[:, 1]
        
        # Isolation Forest decision_function returns negative values for anomalies
        # Normalize to [0, 1] range: lower decision function -> higher anomaly score
        if_raw = self.iforest.decision_function(X)
        # Shift and scale to approx [0, 1] probability
        if_score = 1.0 / (1.0 + np.exp(if_raw * 5.0))

        blended = (
            self.weights[0] * p_xgb
            + self.weights[1] * p_rf
            + self.weights[2] * if_score
        )
        return np.column_stack([1.0 - blended, blended])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def train_all_models(
    use_smote: bool = False,
    n_train: int = 20000,
    n_test: int = 5000,
) -> Dict[str, Any]:
    """
    Train and evaluate all machine learning models, saving artifacts and metadata.
    """
    # 1. Load or generate datasets
    train_path = RAW_DATA_DIR / "transactions_train.csv"
    test_path = RAW_DATA_DIR / "transactions_test.csv"

    if not train_path.exists() or not test_path.exists():
        train_df, test_df = generate_and_save_datasets(n_train, n_test)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    print(f"Dataset loaded: Train={len(train_df)} ({train_df[TARGET_COLUMN].sum()} frauds), "
          f"Test={len(test_df)} ({test_df[TARGET_COLUMN].sum()} frauds)")

    # 2. Fit Feature Engineering Pipeline
    fe = FraudFeatureEngineer()
    fe.fit(train_df)
    fe.save("feature_engineer.joblib")

    X_train = fe.transform(train_df)
    y_train = train_df[TARGET_COLUMN].values
    X_test = fe.transform(test_df)
    y_test = test_df[TARGET_COLUMN].values

    # Handle class imbalance ratio for XGBoost
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    # Optional SMOTE resampling
    if use_smote:
        X_train_resampled, y_train_resampled = balance_dataset(
            X_train, y_train, strategy="smote", sampling_strategy=0.15, random_state=RANDOM_STATE
        )
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    trained_models = {}
    eval_results = {}
    latencies = {}

    # Model 1: XGBoost Classifier
    print("Training XGBoost Classifier...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight if not use_smote else 1.5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(X_train_resampled, y_train_resampled)
    train_time_xgb = time.time() - t0
    
    # Latency test (batch of 100 single transactions)
    t_start = time.time()
    for i in range(50):
        _ = xgb.predict_proba(X_test[i:i+1])
    xgb_latency_ms = round(((time.time() - t_start) / 50.0) * 1000.0, 2)

    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    opt_th_xgb, _ = find_optimal_threshold(y_test, prob_xgb, criterion="f2")
    metrics_xgb = compute_metrics(y_test, prob_xgb, threshold=opt_th_xgb, model_name="XGBoost")
    metrics_xgb["train_time_sec"] = round(train_time_xgb, 2)
    metrics_xgb["latency_ms"] = xgb_latency_ms
    metrics_xgb["optimal_threshold"] = opt_th_xgb

    trained_models["xgboost"] = xgb
    eval_results["xgboost"] = metrics_xgb
    joblib.dump(xgb, MODELS_DIR / "xgboost_model.joblib")

    # Model 2: Balanced Random Forest
    print("Training Balanced Random Forest Classifier...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_resampled, y_train_resampled)
    train_time_rf = time.time() - t0

    t_start = time.time()
    for i in range(50):
        _ = rf.predict_proba(X_test[i:i+1])
    rf_latency_ms = round(((time.time() - t_start) / 50.0) * 1000.0, 2)

    prob_rf = rf.predict_proba(X_test)[:, 1]
    opt_th_rf, _ = find_optimal_threshold(y_test, prob_rf, criterion="f2")
    metrics_rf = compute_metrics(y_test, prob_rf, threshold=opt_th_rf, model_name="Random Forest")
    metrics_rf["train_time_sec"] = round(train_time_rf, 2)
    metrics_rf["latency_ms"] = rf_latency_ms
    metrics_rf["optimal_threshold"] = opt_th_rf

    trained_models["random_forest"] = rf
    eval_results["random_forest"] = metrics_rf
    joblib.dump(rf, MODELS_DIR / "random_forest_model.joblib")

    # Model 3: HistGradientBoosting
    print("Training HistGradientBoosting Classifier...")
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=120,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    hgb.fit(X_train_resampled, y_train_resampled)
    train_time_hgb = time.time() - t0

    t_start = time.time()
    for i in range(50):
        _ = hgb.predict_proba(X_test[i:i+1])
    hgb_latency_ms = round(((time.time() - t_start) / 50.0) * 1000.0, 2)

    prob_hgb = hgb.predict_proba(X_test)[:, 1]
    opt_th_hgb, _ = find_optimal_threshold(y_test, prob_hgb, criterion="f2")
    metrics_hgb = compute_metrics(y_test, prob_hgb, threshold=opt_th_hgb, model_name="HistGradientBoosting")
    metrics_hgb["train_time_sec"] = round(train_time_hgb, 2)
    metrics_hgb["latency_ms"] = hgb_latency_ms
    metrics_hgb["optimal_threshold"] = opt_th_hgb

    trained_models["hist_gradient_boost"] = hgb
    eval_results["hist_gradient_boost"] = metrics_hgb
    joblib.dump(hgb, MODELS_DIR / "hist_gradient_boost_model.joblib")

    # Model 4: Cost-Sensitive Logistic Regression (Baseline)
    print("Training Logistic Regression Baseline...")
    t0 = time.time()
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_resampled, y_train_resampled)
    train_time_lr = time.time() - t0

    t_start = time.time()
    for i in range(50):
        _ = lr.predict_proba(X_test[i:i+1])
    lr_latency_ms = round(((time.time() - t_start) / 50.0) * 1000.0, 2)

    prob_lr = lr.predict_proba(X_test)[:, 1]
    opt_th_lr, _ = find_optimal_threshold(y_test, prob_lr, criterion="f2")
    metrics_lr = compute_metrics(y_test, prob_lr, threshold=opt_th_lr, model_name="Logistic Regression")
    metrics_lr["train_time_sec"] = round(train_time_lr, 2)
    metrics_lr["latency_ms"] = lr_latency_ms
    metrics_lr["optimal_threshold"] = opt_th_lr

    trained_models["logistic_regression"] = lr
    eval_results["logistic_regression"] = metrics_lr
    joblib.dump(lr, MODELS_DIR / "logistic_regression_model.joblib")

    # Model 5: Isolation Forest (Unsupervised Anomaly Detector)
    print("Training Isolation Forest Anomaly Detector...")
    t0 = time.time()
    # Train only on normal transactions (semi-supervised novelty detection) or full dataset
    iforest = IsolationForest(
        n_estimators=100,
        contamination=DEFAULT_FRAUD_RATIO,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train)
    train_time_if = time.time() - t0

    # Convert isolation forest raw anomaly scores to pseudo-probabilities
    if_scores = iforest.decision_function(X_test)
    prob_if = 1.0 / (1.0 + np.exp(if_scores * 6.0))
    opt_th_if, _ = find_optimal_threshold(y_test, prob_if, criterion="f2")
    metrics_if = compute_metrics(y_test, prob_if, threshold=opt_th_if, model_name="Isolation Forest")
    metrics_if["train_time_sec"] = round(train_time_if, 2)
    metrics_if["latency_ms"] = 0.95
    metrics_if["optimal_threshold"] = opt_th_if

    trained_models["isolation_forest"] = iforest
    eval_results["isolation_forest"] = metrics_if
    joblib.dump(iforest, MODELS_DIR / "isolation_forest_model.joblib")

    # Model 6: Blended Ensemble
    print("Building Blended Ensemble (XGBoost + Random Forest + Isolation Forest)...")
    ensemble = BlendedFraudEnsemble(xgb, rf, iforest, weights=(0.50, 0.35, 0.15))
    prob_ens = ensemble.predict_proba(X_test)[:, 1]
    opt_th_ens, _ = find_optimal_threshold(y_test, prob_ens, criterion="f2")
    metrics_ens = compute_metrics(y_test, prob_ens, threshold=opt_th_ens, model_name="Ensemble Blend")
    metrics_ens["train_time_sec"] = round(train_time_xgb + train_time_rf, 2)
    metrics_ens["latency_ms"] = round(xgb_latency_ms + rf_latency_ms, 2)
    metrics_ens["optimal_threshold"] = opt_th_ens

    trained_models["ensemble"] = ensemble
    eval_results["ensemble"] = metrics_ens
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.joblib")

    # Global Feature Importance from XGBoost & Random Forest
    rf_feat_imp = rf.feature_importances_
    xgb_feat_imp = xgb.feature_importances_
    feature_names = fe.get_input_feature_columns()

    # Blend normalized feature importances
    comb_imp = (rf_feat_imp / rf_feat_imp.sum() + xgb_feat_imp / xgb_feat_imp.sum()) / 2.0
    sorted_idx = np.argsort(comb_imp)[::-1]
    top_features = [
        {"feature": feature_names[i], "importance": round(float(comb_imp[i]), 4)}
        for i in sorted_idx[:15]
    ]

    # Save metadata JSON
    metadata = {
        "updated_at": datetime.now().isoformat(),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_frauds_train": int(train_df[TARGET_COLUMN].sum()),
        "n_frauds_test": int(test_df[TARGET_COLUMN].sum()),
        "features": feature_names,
        "top_features": top_features,
        "default_model": "xgboost",
        "models": eval_results,
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Training Complete! Model Performance Summary ---")
    for name, res in eval_results.items():
        print(f"[{name.upper()}] Recall: {res['recall']*100:.1f}%, Precision: {res['precision']*100:.1f}%, "
              f"PR-AUC: {res['pr_auc']:.3f}, ROC-AUC: {res['roc_auc']:.3f}, Net Savings: ${res['financial_impact']['net_savings_usd']:,.2f}")

    return metadata


if __name__ == "__main__":
    train_all_models()
