"""Reusable risk scoring helpers."""

from __future__ import annotations


def probability_to_risk_score(probability: float) -> int:
    """Convert fraud probability into a 0-100 integer risk score."""
    clipped = min(max(float(probability), 0.0), 1.0)
    return int(round(clipped * 100))


def risk_category(risk_score: float) -> str:
    """Return Low, Medium, or High category for a numeric risk score."""
    score = float(risk_score)
    if score <= 30:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


def risk_badge(score: float) -> dict[str, str | int]:
    """Return a compact badge payload for dashboards and APIs."""
    numeric = int(round(float(score)))
    return {"risk_score": numeric, "risk_level": risk_category(numeric)}

