"""Data loading, synthetic data generation, and UPI feature engineering."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from app.config import CATEGORICAL_FEATURES, DATA_DIR, DEFAULT_SYNTHETIC_ROWS, RANDOM_STATE


MERCHANT_CATEGORIES = [
    "grocery",
    "fuel",
    "food_delivery",
    "travel",
    "gaming",
    "utilities",
    "electronics",
    "p2p",
]

PAYMENT_INSTRUMENTS = ["upi_collect", "upi_intent", "qr", "mandate", "p2p"]
CITIES = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"]


def ensure_directories() -> None:
    """Create local data and model directories when they are missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_transactions(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load transactions from CSV or generate a reproducible synthetic dataset.

    The original notebook used a local PaySim CSV that is not committed to the
    repository. This function keeps the app runnable by falling back to
    synthetic UPI-like transactions with documented behavioral risk signals.
    """
    ensure_directories()
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            return add_upi_features(pd.read_csv(path))

    demo_path = DATA_DIR / "synthetic_upi_transactions.csv"
    if demo_path.exists():
        return pd.read_csv(demo_path)

    data = generate_synthetic_upi_transactions(DEFAULT_SYNTHETIC_ROWS)
    data.to_csv(demo_path, index=False)
    return data


def generate_synthetic_upi_transactions(rows: int = DEFAULT_SYNTHETIC_ROWS) -> pd.DataFrame:
    """Generate UPI-like transactions with realistic fraud-risk drivers.

    Synthetic derivation:
    - Device/location changes are Bernoulli signals with elevated fraud weight.
    - Night transactions are derived from transaction hour.
    - Velocity, failed attempts, device age, and account age are sampled from
      skewed distributions often seen in payment telemetry.
    - Merchant fraud/failure/dispute rates are generated per merchant and then
      joined back to transactions.
    - The fraud label is sampled from a probability score built from these
      signals, preserving the old binary target while enabling risk scoring.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    user_count = max(200, rows // 10)
    merchant_count = max(80, rows // 35)
    users = [f"U{idx:05d}" for idx in range(1, user_count + 1)]
    merchants = [f"M{idx:04d}" for idx in range(1, merchant_count + 1)]

    merchant_base = pd.DataFrame(
        {
            "merchant_id": merchants,
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, merchant_count),
            "merchant_fraud_rate": np.clip(rng.beta(1.8, 18, merchant_count), 0, 0.45),
            "merchant_failure_rate": np.clip(rng.beta(2.2, 16, merchant_count), 0, 0.5),
            "merchant_dispute_rate": np.clip(rng.beta(1.6, 28, merchant_count), 0, 0.25),
        }
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{idx:08d}" for idx in range(1, rows + 1)],
            "user_id": rng.choice(users, rows),
            "merchant_id": rng.choice(merchants, rows),
            "amount": np.round(rng.lognormal(mean=6.35, sigma=1.05, size=rows), 2),
            "transaction_hour": rng.integers(0, 24, rows),
            "payment_instrument": rng.choice(PAYMENT_INSTRUMENTS, rows),
            "device_changed": rng.binomial(1, 0.13, rows),
            "location_changed": rng.binomial(1, 0.16, rows),
            "failed_attempts": np.clip(rng.poisson(0.55, rows), 0, 8),
            "transactions_last_hour": np.clip(rng.poisson(1.6, rows), 0, 20),
            "device_age_days": np.clip(rng.gamma(shape=2.4, scale=90, size=rows), 1, 1200).astype(int),
            "account_age_days": np.clip(rng.gamma(shape=3.3, scale=120, size=rows), 1, 2500).astype(int),
            "city": rng.choice(CITIES, rows),
            "created_at": [
                (now - timedelta(minutes=int(x))).isoformat()
                for x in rng.integers(0, 60 * 24 * 21, rows)
            ],
        }
    )
    df = df.merge(merchant_base, on="merchant_id", how="left")
    df["night_transaction"] = df["transaction_hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)

    category_multiplier = df["merchant_category"].map(
        {
            "gaming": 1.45,
            "electronics": 1.35,
            "travel": 1.15,
            "p2p": 1.2,
            "grocery": 0.7,
            "utilities": 0.65,
            "fuel": 0.8,
            "food_delivery": 0.9,
        }
    )
    user_mean = df.groupby("user_id")["amount"].transform("mean").clip(lower=1)
    df["avg_amount_deviation"] = np.clip(df["amount"] / user_mean, 0, 15)
    df["user_velocity_score"] = np.clip((df["transactions_last_hour"] / 8) * 100, 0, 100)

    logit = (
        -4.0
        + 0.004 * np.minimum(df["amount"], 20_000)
        + 1.0 * df["device_changed"]
        + 0.9 * df["location_changed"]
        + 0.75 * df["night_transaction"]
        + 0.38 * df["failed_attempts"]
        + 0.18 * df["transactions_last_hour"]
        + 0.35 * np.maximum(df["avg_amount_deviation"] - 2.5, 0)
        + 5.5 * df["merchant_fraud_rate"]
        + 2.8 * df["merchant_failure_rate"]
        + 2.0 * df["merchant_dispute_rate"]
        - 0.0012 * np.minimum(df["account_age_days"], 900)
        - 0.001 * np.minimum(df["device_age_days"], 700)
    ) * category_multiplier
    probability = 1 / (1 + np.exp(-logit))
    df["fraud_probability_seed"] = np.clip(probability, 0.002, 0.98)
    df["is_fraud"] = rng.binomial(1, df["fraud_probability_seed"])
    df["is_failed"] = rng.binomial(
        1,
        np.clip(0.04 + df["merchant_failure_rate"] + 0.03 * df["failed_attempts"], 0, 0.9),
    )
    df["is_disputed"] = rng.binomial(
        1,
        np.clip(0.01 + df["merchant_dispute_rate"] + 0.18 * df["is_fraud"], 0, 0.75),
    )
    return df


def add_upi_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add required UPI risk signals to arbitrary transaction data."""
    df = transactions.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    if "amount" not in df and "oldbalanceOrg" in df and "newbalanceOrig" in df:
        df["amount"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]).abs()
    df["amount"] = pd.to_numeric(df.get("amount", rng.lognormal(6, 1, len(df))), errors="coerce").fillna(0)

    if "transaction_hour" not in df:
        source = pd.to_numeric(df.get("step", rng.integers(0, 24, len(df))), errors="coerce").fillna(0)
        df["transaction_hour"] = (source.astype(int) % 24).astype(int)
    df["night_transaction"] = df["transaction_hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)

    for column, probability in {
        "device_changed": 0.12,
        "location_changed": 0.15,
        "is_failed": 0.08,
        "is_disputed": 0.03,
    }.items():
        if column not in df:
            df[column] = rng.binomial(1, probability, len(df))

    defaults = {
        "failed_attempts": rng.poisson(0.45, len(df)),
        "transactions_last_hour": rng.poisson(1.5, len(df)),
        "device_age_days": rng.integers(1, 900, len(df)),
        "account_age_days": rng.integers(1, 1800, len(df)),
        "user_id": [f"U{idx % 500:05d}" for idx in range(len(df))],
        "merchant_id": [f"M{idx % 120:04d}" for idx in range(len(df))],
        "transaction_id": [f"TXN{idx:08d}" for idx in range(len(df))],
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, len(df)),
        "payment_instrument": rng.choice(PAYMENT_INSTRUMENTS, len(df)),
    }
    for column, value in defaults.items():
        if column not in df:
            df[column] = value

    user_mean = df.groupby("user_id")["amount"].transform("mean").clip(lower=1)
    if "avg_amount_deviation" not in df:
        df["avg_amount_deviation"] = np.clip(df["amount"] / user_mean, 0, 15)
    if "user_velocity_score" not in df:
        df["user_velocity_score"] = np.clip((df["transactions_last_hour"] / 8) * 100, 0, 100)

    if "is_fraud" not in df and "isFraud" in df:
        df["is_fraud"] = df["isFraud"]
    if "is_fraud" not in df:
        df["is_fraud"] = 0

    merchant_metrics = (
        df.groupby("merchant_id")
        .agg(
            merchant_fraud_rate=("is_fraud", "mean"),
            merchant_failure_rate=("is_failed", "mean"),
            merchant_dispute_rate=("is_disputed", "mean"),
        )
        .reset_index()
    )
    for column in ["merchant_fraud_rate", "merchant_failure_rate", "merchant_dispute_rate"]:
        if column in df:
            df = df.drop(columns=[column])
    df = df.merge(merchant_metrics, on="merchant_id", how="left")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str)
    return df


def top_numeric_correlations(df: pd.DataFrame, target: str, n: int = 8) -> pd.Series:
    """Return the strongest absolute numeric correlations with a target column."""
    numeric = df.select_dtypes(include=["number"])
    if target not in numeric:
        return pd.Series(dtype=float)
    return numeric.corr(numeric_only=True)[target].drop(target, errors="ignore").abs().sort_values(ascending=False).head(n)


def format_percentage(value: float) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.1f}%"


def filter_by_options(df: pd.DataFrame, column: str, selected: Iterable[str]) -> pd.DataFrame:
    """Filter a DataFrame by selected categorical options."""
    options = list(selected)
    if not options:
        return df
    return df[df[column].isin(options)]
