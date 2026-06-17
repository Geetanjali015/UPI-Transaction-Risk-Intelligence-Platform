"""Explainability utilities using SHAP when available with graceful fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import ALL_MODEL_FEATURES
from app.ml.predict import load_model_bundle, prepare_transaction


def feature_names(bundle: dict[str, Any] | None = None) -> list[str]:
    """Return transformed feature names from the model preprocessor."""
    bundle = bundle or load_model_bundle()
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        return list(bundle.get("features", ALL_MODEL_FEATURES))


def feature_importance() -> pd.DataFrame:
    """Return model feature importances for analytics views."""
    bundle = load_model_bundle()
    model = bundle["pipeline"].named_steps["model"]
    names = feature_names(bundle)
    importances = getattr(model, "feature_importances_", np.zeros(len(names)))
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_transaction(transaction: dict[str, Any], limit: int = 8) -> dict[str, Any]:
    """Explain one transaction using SHAP if installed, otherwise importance-weighted signals."""
    bundle = load_model_bundle()
    prepared = prepare_transaction(transaction)
    x = pd.DataFrame([{feature: prepared[feature] for feature in ALL_MODEL_FEATURES}])
    transformed = bundle["pipeline"].named_steps["preprocessor"].transform(x)
    names = feature_names(bundle)

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(bundle["pipeline"].named_steps["model"])
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            class_values = values[1][0]
            base_value = explainer.expected_value[1]
        else:
            class_values = values[0, :, 1] if values.ndim == 3 else values[0]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        contributions = _rank_contributions(names, class_values, limit)
        return {
            "method": "shap",
            "base_value": float(base_value),
            "top_contributions": contributions,
        }
    except Exception:
        model = bundle["pipeline"].named_steps["model"]
        importances = getattr(model, "feature_importances_", np.ones(len(names)))
        dense = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
        values = dense * importances
        return {
            "method": "feature_importance_fallback",
            "base_value": 0.0,
            "top_contributions": _rank_contributions(names, values, limit),
        }


def _rank_contributions(names: list[str], values: np.ndarray, limit: int) -> list[dict[str, float | str]]:
    """Rank absolute contribution values and return a serializable payload."""
    records = [{"feature": name, "contribution": float(value)} for name, value in zip(names, np.asarray(values).ravel())]
    return sorted(records, key=lambda item: abs(float(item["contribution"])), reverse=True)[:limit]
