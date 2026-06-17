"""FastAPI routes for the UPI risk intelligence platform."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ScoreResponse, TransactionRequest
from app.ml.explainability import explain_transaction, feature_importance
from app.ml.predict import model_info, score_transaction
from app.utils.helpers import load_transactions

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "upi-risk-intelligence"}


@router.post("/score", response_model=ScoreResponse)
def score(request: TransactionRequest) -> dict:
    """Score a transaction and return the authorization decision."""
    result = score_transaction(request.dict())
    return {key: value for key, value in result.items() if key != "transaction"}


@router.post("/simulate")
def simulate(request: TransactionRequest) -> dict:
    """Score a transaction and include explainability output."""
    payload = request.dict()
    result = score_transaction(payload)
    result["explanation"] = explain_transaction(payload)
    return result


@router.get("/metrics")
def metrics() -> dict:
    """Return operational and model metrics for monitoring."""
    transactions = load_transactions()
    info = model_info()
    return {
        "transactions": int(len(transactions)),
        "fraud_rate": float(transactions["is_fraud"].mean()),
        "failure_rate": float(transactions["is_failed"].mean()),
        "dispute_rate": float(transactions["is_disputed"].mean()),
        "model_metrics": info.get("metrics", {}),
    }


@router.get("/model-info")
def get_model_info() -> dict:
    """Return active model metadata."""
    return model_info()


@router.get("/feature-importance")
def get_feature_importance() -> list[dict]:
    """Return feature importance ranking."""
    return feature_importance().head(25).to_dict("records")

