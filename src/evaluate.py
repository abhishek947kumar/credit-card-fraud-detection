"""
Evaluation and Metrics Engine for Fraud Detection Models.
Computes Precision, Recall, F1, F2, PR-AUC, ROC-AUC, Confusion Matrix,
and Financial Business Impact (dollar losses prevented vs analyst review costs).
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud detection evaluation metrics.
    
    Args:
        y_true: Ground truth binary labels (0 = legitimate, 1 = fraud)
        y_prob: Predicted fraud probabilities [0.0, 1.0]
        threshold: Decision threshold for binary classification
        model_name: Identifier for the model
        
    Returns:
        Dictionary containing statistical metrics and financial impact metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    total = len(y_true)
    total_fraud = int(np.sum(y_true))
    total_legit = total - total_fraud

    # Core ML Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    
    # AUC metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    # Business / Financial Impact
    # Without any ML system: all frauds go through as False Negatives
    baseline_loss = total_fraud * COST_FALSE_NEGATIVE
    # With ML system:
    # Missed frauds cost COST_FALSE_NEGATIVE
    # False alarms (friction/review) cost COST_FALSE_POSITIVE
    system_loss = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
    net_fraud_prevented = (tp * COST_FALSE_NEGATIVE) - (fp * COST_FALSE_POSITIVE)
    cost_reduction_pct = ((baseline_loss - system_loss) / max(baseline_loss, 1.0)) * 100.0

    # Downsampled curves for UI visualization
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)

    # Subsample curves to ~50 points for clean, fast frontend rendering
    roc_points = _subsample_curve(fpr, tpr, max_points=40)
    pr_points = _subsample_curve(recalls, precisions, max_points=40)

    return {
        "model_name": model_name,
        "threshold": threshold,
        "total_samples": total,
        "total_frauds": total_fraud,
        "total_legitimate": total_legit,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "financial_impact": {
            "baseline_loss_usd": round(baseline_loss, 2),
            "system_loss_usd": round(system_loss, 2),
            "net_savings_usd": round(net_fraud_prevented, 2),
            "cost_reduction_percent": round(cost_reduction_pct, 1),
            "fraud_prevented_count": int(tp),
            "fraud_missed_count": int(fn),
            "false_alarms_count": int(fp),
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "f2",
) -> Tuple[float, float]:
    """
    Find optimal classification threshold balancing Precision and Recall
    or minimizing dollar costs.
    
    criterion: 'f1', 'f2', or 'cost'
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_val = -1e9 if criterion != "cost" else 1e9

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        
        if criterion == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_val:
                best_val = score
                best_thresh = th
        elif criterion == "f2":
            score = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
            if score > best_val:
                best_val = score
                best_thresh = th
        elif criterion == "cost":
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
            if cost < best_val:
                best_val = cost
                best_thresh = th

    return round(float(best_thresh), 2), round(float(best_val), 4)


def _subsample_curve(x_vals: np.ndarray, y_vals: np.ndarray, max_points: int = 40) -> List[Dict[str, float]]:
    """Subsample 1D arrays to evenly spaced points for JSON/UI."""
    if len(x_vals) <= max_points:
        indices = np.arange(len(x_vals))
    else:
        indices = np.linspace(0, len(x_vals) - 1, max_points, dtype=int)
        
    return [{"x": round(float(x_vals[i]), 4), "y": round(float(y_vals[i]), 4)} for i in indices]
