"""Production-style payment decision rules layered over ML risk scores."""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.risk_scoring import risk_category


@dataclass(frozen=True)
class DecisionResult:
    """Decision response produced by the risk engine."""

    decision: str
    reason: str
    risk_score: int
    risk_category: str
    top_risk_factors: list[str]


def high_risk_factors(transaction: dict, limit: int = 5) -> list[str]:
    """Infer readable top risk factors from engineered transaction signals."""
    candidates: list[tuple[str, float]] = [
        ("device_changed", 22.0 if int(transaction.get("device_changed", 0)) else 0.0),
        ("location_changed", 18.0 if int(transaction.get("location_changed", 0)) else 0.0),
        ("night_transaction", 16.0 if int(transaction.get("night_transaction", 0)) else 0.0),
        ("failed_attempts", min(float(transaction.get("failed_attempts", 0)) * 8.0, 30.0)),
        (
            "transactions_last_hour",
            min(float(transaction.get("transactions_last_hour", 0)) * 4.0, 28.0),
        ),
        (
            "avg_amount_deviation",
            max(float(transaction.get("avg_amount_deviation", 0)) - 1.0, 0.0) * 9.0,
        ),
        ("merchant_fraud_rate", float(transaction.get("merchant_fraud_rate", 0)) * 100.0),
        ("merchant_failure_rate", float(transaction.get("merchant_failure_rate", 0)) * 60.0),
        ("merchant_dispute_rate", float(transaction.get("merchant_dispute_rate", 0)) * 80.0),
        ("user_velocity_score", float(transaction.get("user_velocity_score", 0)) * 0.4),
        (
            "new_device",
            18.0 if float(transaction.get("device_age_days", 999)) <= 14 else 0.0,
        ),
        (
            "new_account",
            18.0 if float(transaction.get("account_age_days", 999)) <= 30 else 0.0,
        ),
    ]
    ranked = [name for name, score in sorted(candidates, key=lambda item: item[1], reverse=True) if score > 0]
    return ranked[:limit]


def decide_transaction(risk_score: int, transaction: dict | None = None) -> DecisionResult:
    """Apply deterministic payment rules to a risk score.

    Rules:
    - risk score below 30: ALLOW
    - risk score from 30 to below 70: OTP_VERIFICATION
    - risk score 70 and above: BLOCK
    """
    score = int(round(risk_score))
    factors = high_risk_factors(transaction or {})
    category = risk_category(score)

    if score < 30:
        return DecisionResult(
            decision="ALLOW",
            reason="Risk score is below the operational review threshold.",
            risk_score=score,
            risk_category=category,
            top_risk_factors=factors,
        )
    if score < 70:
        return DecisionResult(
            decision="OTP_VERIFICATION",
            reason="Moderate risk requires step-up authentication before payment completion.",
            risk_score=score,
            risk_category=category,
            top_risk_factors=factors,
        )
    return DecisionResult(
        decision="BLOCK",
        reason="High-risk pattern exceeds the blocking threshold for real-time authorization.",
        risk_score=score,
        risk_category=category,
        top_risk_factors=factors,
    )

