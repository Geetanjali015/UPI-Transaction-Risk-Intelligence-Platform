"""Merchant-level risk profile generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.risk_scoring import risk_category


def build_merchant_profiles(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate merchant fraud, failure, dispute, and volume risk metrics."""
    grouped = (
        transactions.groupby(["merchant_id", "merchant_category"])
        .agg(
            merchant_transaction_volume=("transaction_id", "count"),
            merchant_average_amount=("amount", "mean"),
            merchant_fraud_rate=("is_fraud", "mean"),
            merchant_failure_rate=("is_failed", "mean"),
            merchant_dispute_rate=("is_disputed", "mean"),
        )
        .reset_index()
    )
    volume_component = np.clip(grouped["merchant_transaction_volume"] / 120 * 12, 0, 12)
    amount_component = np.clip(grouped["merchant_average_amount"] / 25_000 * 10, 0, 10)
    fraud_component = np.clip(grouped["merchant_fraud_rate"] * 55, 0, 55)
    failure_component = np.clip(grouped["merchant_failure_rate"] * 25, 0, 25)
    dispute_component = np.clip(grouped["merchant_dispute_rate"] * 30, 0, 30)

    grouped["merchant_risk_score"] = np.clip(
        volume_component + amount_component + fraud_component + failure_component + dispute_component,
        0,
        100,
    ).round(0).astype(int)
    grouped["merchant_risk_level"] = grouped["merchant_risk_score"].apply(risk_category)
    return grouped.sort_values("merchant_risk_score", ascending=False)


def score_merchant_profile(transaction: dict) -> int:
    """Score a single transaction's merchant context."""
    score = (
        float(transaction.get("merchant_fraud_rate", 0)) * 55
        + float(transaction.get("merchant_failure_rate", 0)) * 25
        + float(transaction.get("merchant_dispute_rate", 0)) * 30
    )
    return int(round(min(max(score, 0), 100)))
