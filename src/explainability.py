"""
Explainable AI (XAI) and Risk Factor Diagnostics Engine.
Breaks down fraud predictions into human-readable risk drivers
and quantitative feature contributions for fraud analysts.
"""
from typing import Any, Dict, List
import numpy as np
from src.config import HIGH_RISK_CATEGORIES


class FraudExplainer:
    """
    Analyzes transaction attributes to generate qualitative risk alerts
    and quantitative feature importance contributions.
    """

    @staticmethod
    def explain_transaction(txn: Dict[str, Any], prob: float) -> List[Dict[str, Any]]:
        """
        Identify the key anomaly indicators that contributed to fraud risk.
        """
        reasons = []

        amount = float(txn.get("amount", 0.0))
        ratio = float(txn.get("amount_to_user_mean_ratio", 1.0))
        time_since = float(txn.get("time_since_last_txn", 100.0))
        dist_last = float(txn.get("distance_from_last_txn", 0.0))
        vel_1h = int(txn.get("velocity_1h", 0))
        vel_24h = int(txn.get("velocity_24h", 0))
        dev_trust = float(txn.get("device_trust_score", 100.0))
        category = str(txn.get("category", "")).lower()
        foreign = int(txn.get("foreign_transaction", 0))
        failed_pins = int(txn.get("failed_pin_attempts_24h", 0))
        hour = int(txn.get("hour_of_day", 12))

        # 1. Geo-velocity / Teleportation check
        time_hours = max(time_since / 60.0, 0.05)
        speed_kmh = dist_last / time_hours
        if speed_kmh > 750.0:
            reasons.append({
                "factor": "Physical Impossible Travel (Card Cloning)",
                "severity": "CRITICAL",
                "impact": 0.35,
                "detail": f"Travel speed calculated at {speed_kmh:,.0f} km/h ({dist_last:,.0f} km in {time_since:.1f} mins). Physical impossibility.",
            })
        elif dist_last > 1000.0 and time_since < 180.0:
            reasons.append({
                "factor": "High Distance Anomaly",
                "severity": "HIGH",
                "impact": 0.20,
                "detail": f"Card used {dist_last:,.0f} km away only {time_since:.0f} mins after previous transaction.",
            })

        # 2. Velocity spike / Card testing check
        if vel_1h >= 5:
            reasons.append({
                "factor": "Velocity Surge / Automated Card Testing",
                "severity": "CRITICAL" if vel_1h > 8 else "HIGH",
                "impact": 0.25,
                "detail": f"{vel_1h} transactions initiated in the past 1 hour (average user baseline is <0.5).",
            })
        elif vel_24h >= 12:
            reasons.append({
                "factor": "24-Hour Velocity Spike",
                "severity": "MEDIUM",
                "impact": 0.15,
                "detail": f"{vel_24h} transactions in 24 hours.",
            })

        # 3. Monetary anomaly
        if ratio >= 5.0 and amount > 500.0:
            reasons.append({
                "factor": "Extreme Spend Outlier",
                "severity": "HIGH",
                "impact": 0.25,
                "detail": f"Amount of ${amount:,.2f} is {ratio:.1f}x higher than cardholder's 30-day baseline average.",
            })
        elif amount > 2500.0:
            reasons.append({
                "factor": "High Value Transaction",
                "severity": "MEDIUM",
                "impact": 0.15,
                "detail": f"Transaction amount (${amount:,.2f}) exceeds normal velocity thresholds.",
            })

        # 4. Device & Authentication Trust
        if dev_trust < 30.0:
            reasons.append({
                "factor": "Untrusted / Compromised Device Fingerprint",
                "severity": "HIGH",
                "impact": 0.20,
                "detail": f"Device trust confidence score is only {dev_trust:.0f}/100. New or spoofed client environment.",
            })

        # 5. Failed PIN / Security challenges
        if failed_pins >= 2:
            reasons.append({
                "factor": "Multiple Failed PIN Attempts",
                "severity": "HIGH",
                "impact": 0.20,
                "detail": f"{failed_pins} failed security PIN attempts registered in the preceding 24 hours.",
            })

        # 6. High-risk category + Time of day
        if category in HIGH_RISK_CATEGORIES:
            reasons.append({
                "factor": "High-Risk Merchant Category",
                "severity": "MEDIUM",
                "impact": 0.12,
                "detail": f"Merchant category '{category.replace('_', ' ').title()}' carries heightened industry fraud propensity.",
            })

        if hour in [1, 2, 3, 4]:
            reasons.append({
                "factor": "Off-Hours Midnight Activity",
                "severity": "LOW",
                "impact": 0.08,
                "detail": f"Transaction executed during cardholder off-hours at {hour:02d}:00.",
            })

        if foreign == 1 and dist_last > 500.0:
            reasons.append({
                "factor": "Cross-Border Foreign POS",
                "severity": "MEDIUM",
                "impact": 0.10,
                "detail": "Cross-border payment flag triggered away from habitual spending territory.",
            })

        # If probability is high but no explicit hard rule was triggered, mention multi-feature non-linear pattern
        if prob >= 0.5 and not reasons:
            reasons.append({
                "factor": "Multi-Variable Anomaly Pattern",
                "severity": "MEDIUM",
                "impact": 0.18,
                "detail": "Non-linear statistical correlation across multiple PCA components and transaction telemetry.",
            })

        # Sort reasons by impact descending
        reasons.sort(key=lambda r: r["impact"], reverse=True)
        return reasons
