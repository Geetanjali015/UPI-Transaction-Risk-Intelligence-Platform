"""Central configuration for the UPI risk intelligence platform."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"

RANDOM_STATE = 42
DEFAULT_SYNTHETIC_ROWS = 5_000

RISK_FEATURES = [
    "amount",
    "transaction_hour",
    "device_changed",
    "location_changed",
    "night_transaction",
    "failed_attempts",
    "transactions_last_hour",
    "avg_amount_deviation",
    "merchant_fraud_rate",
    "merchant_failure_rate",
    "merchant_dispute_rate",
    "user_velocity_score",
    "device_age_days",
    "account_age_days",
]

CATEGORICAL_FEATURES = ["merchant_category", "payment_instrument"]

ALL_MODEL_FEATURES = RISK_FEATURES + CATEGORICAL_FEATURES

DECISION_COLORS = {
    "ALLOW": "#0f9f6e",
    "OTP_VERIFICATION": "#d97706",
    "BLOCK": "#dc2626",
}

RISK_COLORS = {
    "Low": "#0f9f6e",
    "Medium": "#d97706",
    "High": "#dc2626",
}

