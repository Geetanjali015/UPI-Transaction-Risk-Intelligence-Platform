"""Prediction helpers for API and dashboard use."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import ALL_MODEL_FEATURES, MODEL_PATH
from app.ml.risk_scoring import probability_to_risk_score, risk_category
from app.ml.train import train_model
from app.profiling.merchant_profile import score_merchant_profile
from app.profiling.user_profile import score_user_profile
from app.rules.decision_engine import decide_transaction


DEFAULT_TRANSACTION = {
    "amount": 1_250.0,
    "transaction_hour": 14,
    "merchant_category": "grocery",
    "payment_instrument": "upi_intent",
    "device_changed": 0,
    "location_changed": 0,
    "night_transaction": 0,
    "failed_attempts": 0,
    "transactions_last_hour": 1,
    "avg_amount_deviation": 1.0,
    "merchant_fraud_rate": 0.03,
    "merchant_failure_rate": 0.06,
    "merchant_dispute_rate": 0.01,
    "user_velocity_score": 12.5,
    "device_age_days": 180,
    "account_age_days": 420,
}


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    """Load a persisted model bundle, training a demo model if necessary."""
    if not MODEL_PATH.exists():
        return train_model()
    return joblib.load(MODEL_PATH)


def refresh_model_bundle() -> dict[str, Any]:
    """Retrain and reload the cached model bundle."""
    load_model_bundle.cache_clear()
    bundle = train_model()
    load_model_bundle.cache_clear()
    return bundle


def prepare_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    """Fill missing fields and derive dependent features for one transaction."""
    prepared = {**DEFAULT_TRANSACTION, **transaction}
    prepared["amount"] = float(prepared["amount"])
    prepared["transaction_hour"] = int(prepared["transaction_hour"]) % 24
    prepared["night_transaction"] = int(prepared.get("night_transaction") or prepared["transaction_hour"] in [0, 1, 2, 3, 4, 23])
    prepared["device_changed"] = int(bool(prepared.get("device_changed", 0)))
    prepared["location_changed"] = int(bool(prepared.get("location_changed", 0)))
    prepared["failed_attempts"] = int(prepared.get("failed_attempts", 0))
    prepared["transactions_last_hour"] = int(prepared.get("transactions_last_hour", 1))
    prepared["avg_amount_deviation"] = float(prepared.get("avg_amount_deviation", 1.0))
    prepared["merchant_fraud_rate"] = float(prepared.get("merchant_fraud_rate", 0.03))
    prepared["merchant_failure_rate"] = float(prepared.get("merchant_failure_rate", 0.06))
    prepared["merchant_dispute_rate"] = float(prepared.get("merchant_dispute_rate", 0.01))
    prepared["user_velocity_score"] = float(
        prepared.get("user_velocity_score", min(prepared["transactions_last_hour"] / 8 * 100, 100))
    )
    prepared["device_age_days"] = int(prepared.get("device_age_days", 180))
    prepared["account_age_days"] = int(prepared.get("account_age_days", 420))
    prepared["merchant_category"] = str(prepared.get("merchant_category", "grocery"))
    prepared["payment_instrument"] = str(prepared.get("payment_instrument", "upi_intent"))
    return prepared


def score_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    """Score one transaction and return a decision-engine payload."""
    bundle = load_model_bundle()
    prepared = prepare_transaction(transaction)
    row = pd.DataFrame([{feature: prepared[feature] for feature in ALL_MODEL_FEATURES}])
    probability = float(bundle["pipeline"].predict_proba(row)[:, 1][0])
    score = probability_to_risk_score(probability)
    decision = decide_transaction(score, prepared)
    return {
        "fraud_probability": round(probability, 4),
        "risk_score": score,
        "risk_level": risk_category(score),
        "decision": decision.decision,
        "reason": decision.reason,
        "top_risk_factors": decision.top_risk_factors,
        "user_risk_score": score_user_profile(prepared),
        "merchant_risk_score": score_merchant_profile(prepared),
        "transaction": prepared,
    }


def score_dataframe(transactions: pd.DataFrame) -> pd.DataFrame:
    """Append fraud probability, risk score, category, and decision to rows."""
    bundle = load_model_bundle()
    rows = [prepare_transaction(row) for row in transactions.to_dict("records")]
    x = pd.DataFrame([{feature: row[feature] for feature in ALL_MODEL_FEATURES} for row in rows])
    probabilities = bundle["pipeline"].predict_proba(x)[:, 1]
    results = transactions.copy()
    results["fraud_probability"] = np.round(probabilities, 4)
    results["risk_score"] = [probability_to_risk_score(prob) for prob in probabilities]
    results["risk_level"] = results["risk_score"].apply(risk_category)
    decisions = [decide_transaction(score, row) for score, row in zip(results["risk_score"], rows)]
    results["decision"] = [item.decision for item in decisions]
    results["decision_reason"] = [item.reason for item in decisions]
    results["top_risk_factors"] = [", ".join(item.top_risk_factors) for item in decisions]
    return results


def model_info() -> dict[str, Any]:
    """Return metadata for the active risk model."""
    bundle = load_model_bundle()
    return {
        "model_name": bundle.get("model_name", "RandomForestClassifier"),
        "trained_at": bundle.get("trained_at"),
        "training_rows": bundle.get("training_rows"),
        "features": bundle.get("features", ALL_MODEL_FEATURES),
        "metrics": bundle.get("metrics", {}),
    }
