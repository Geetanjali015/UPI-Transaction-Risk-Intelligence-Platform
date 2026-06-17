"""Model training entry point for the UPI risk engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import ALL_MODEL_FEATURES, CATEGORICAL_FEATURES, MODEL_DIR, MODEL_PATH, RANDOM_STATE, RISK_FEATURES
from app.utils.helpers import load_transactions


def train_model(transactions: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    """Train and optionally persist a Random Forest fraud probability model."""
    df = transactions.copy() if transactions is not None else load_transactions()
    df = df.dropna(subset=["is_fraud"]).copy()

    x = df[ALL_MODEL_FEATURES]
    y = df["is_fraud"].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", RISK_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=140,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = _build_metrics(y_test, predictions, probabilities)
    bundle = {
        "pipeline": pipeline,
        "features": ALL_MODEL_FEATURES,
        "numeric_features": RISK_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "training_rows": int(len(df)),
        "model_name": "RandomForestClassifier",
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
    return bundle


def _build_metrics(y_true: pd.Series, predictions: pd.Series, probabilities: pd.Series) -> dict[str, Any]:
    """Build serializable model performance metrics for dashboards and API."""
    labels = sorted(set(y_true.tolist()) | set(predictions.tolist()))
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, predictions, labels=labels).tolist()

    if y_true.nunique() > 1:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        pr_auc = float(average_precision_score(y_true, probabilities))
        fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
    else:
        roc_auc = 0.0
        pr_auc = 0.0
        fpr, tpr, roc_thresholds = [], [], []
        precision, recall, pr_thresholds = [], [], []

    return {
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": labels,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in roc_thresholds],
        },
        "precision_recall_curve": {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "thresholds": [float(x) for x in pr_thresholds],
        },
    }


if __name__ == "__main__":
    result = train_model()
    print(f"Saved {result['model_name']} to {MODEL_PATH}")
    print(f"ROC-AUC: {result['metrics']['roc_auc']:.3f}")

