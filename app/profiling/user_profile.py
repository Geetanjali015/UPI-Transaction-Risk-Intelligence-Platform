"""User-level risk profile generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.risk_scoring import risk_category


def build_user_profiles(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behavioral transaction signals into user risk profiles."""
    grouped = (
        transactions.groupby("user_id")
        .agg(
            transaction_count=("transaction_id", "count"),
            average_amount=("amount", "mean"),
            failed_attempts=("failed_attempts", "sum"),
            device_switches=("device_changed", "sum"),
            location_switches=("location_changed", "sum"),
            night_transactions=("night_transaction", "sum"),
            average_velocity=("user_velocity_score", "mean"),
            fraud_rate=("is_fraud", "mean"),
        )
        .reset_index()
    )

    amount_component = np.clip(grouped["average_amount"] / 20_000 * 20, 0, 20)
    frequency_component = np.clip(grouped["transaction_count"] / 80 * 15, 0, 15)
    failure_component = np.clip(grouped["failed_attempts"] / 20 * 18, 0, 18)
    device_component = np.clip(grouped["device_switches"] / 10 * 15, 0, 15)
    night_component = np.clip(grouped["night_transactions"] / 12 * 12, 0, 12)
    velocity_component = np.clip(grouped["average_velocity"] / 100 * 10, 0, 10)
    fraud_component = np.clip(grouped["fraud_rate"] * 35, 0, 35)

    grouped["user_risk_score"] = np.clip(
        amount_component
        + frequency_component
        + failure_component
        + device_component
        + night_component
        + velocity_component
        + fraud_component,
        0,
        100,
    ).round(0).astype(int)
    grouped["user_risk_level"] = grouped["user_risk_score"].apply(risk_category)
    return grouped.sort_values("user_risk_score", ascending=False)


def score_user_profile(transaction: dict) -> int:
    """Score a single transaction's user behavior context."""
    score = (
        min(float(transaction.get("transactions_last_hour", 0)) * 5.0, 25.0)
        + min(float(transaction.get("failed_attempts", 0)) * 8.0, 25.0)
        + (16.0 if int(transaction.get("device_changed", 0)) else 0.0)
        + (12.0 if int(transaction.get("night_transaction", 0)) else 0.0)
        + min(max(float(transaction.get("avg_amount_deviation", 1.0)) - 1.0, 0.0) * 10.0, 22.0)
    )
    return int(round(min(score, 100.0)))

