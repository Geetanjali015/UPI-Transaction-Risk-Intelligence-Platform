"""Pydantic schemas for risk-scoring APIs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Request body for scoring or simulating one transaction."""

    amount: float = Field(default=1250.0, ge=0)
    transaction_hour: int = Field(default=14, ge=0, le=23)
    merchant_category: str = "grocery"
    payment_instrument: str = "upi_intent"
    device_changed: bool = False
    location_changed: bool = False
    failed_attempts: int = Field(default=0, ge=0)
    transactions_last_hour: int = Field(default=1, ge=0)
    avg_amount_deviation: float = Field(default=1.0, ge=0)
    merchant_fraud_rate: float = Field(default=0.03, ge=0, le=1)
    merchant_failure_rate: float = Field(default=0.06, ge=0, le=1)
    merchant_dispute_rate: float = Field(default=0.01, ge=0, le=1)
    user_velocity_score: float = Field(default=12.5, ge=0, le=100)
    device_age_days: int = Field(default=180, ge=0)
    account_age_days: int = Field(default=420, ge=0)


class ScoreResponse(BaseModel):
    """Risk score and decision response."""

    fraud_probability: float
    risk_score: int
    risk_level: str
    decision: str
    reason: str
    top_risk_factors: list[str]
    user_risk_score: int
    merchant_risk_score: int

