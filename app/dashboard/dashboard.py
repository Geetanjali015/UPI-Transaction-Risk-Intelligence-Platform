"""Streamlit fraud operations dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DECISION_COLORS, RISK_COLORS
from app.dashboard.components.risk_widgets import decision_chip, inject_global_styles, metric_card, risk_chip
from app.ml.explainability import explain_transaction, feature_importance
from app.ml.predict import model_info, score_dataframe, score_transaction
from app.profiling.merchant_profile import build_merchant_profiles
from app.profiling.user_profile import build_user_profiles
from app.utils.helpers import MERCHANT_CATEGORIES, PAYMENT_INSTRUMENTS, load_transactions


st.set_page_config(
    page_title="UPI Risk Intelligence",
    page_icon="",
    layout="wide",
)
inject_global_styles()


@st.cache_data(show_spinner=False)
def load_scored_transactions() -> pd.DataFrame:
    """Load demo transactions and append real-time risk outputs."""
    return score_dataframe(load_transactions()).sort_values("risk_score", ascending=False)


@st.cache_data(show_spinner=False)
def load_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cached user and merchant profile views."""
    transactions = load_transactions()
    return build_user_profiles(transactions), build_merchant_profiles(transactions)


def app() -> None:
    """Render the dashboard shell and selected page."""
    st.sidebar.title("UPI Risk Ops")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Transaction Monitor",
            "Risk Analytics",
            "User Profiles",
            "Merchant Profiles",
            "Model Explainability",
            "System Metrics",
            "Transaction Simulator",
        ],
    )
    scored = load_scored_transactions()

    if page == "Overview":
        render_overview(scored)
    elif page == "Transaction Monitor":
        render_transaction_monitor(scored)
    elif page == "Risk Analytics":
        render_risk_analytics(scored)
    elif page == "User Profiles":
        render_user_profiles()
    elif page == "Merchant Profiles":
        render_merchant_profiles()
    elif page == "Model Explainability":
        render_explainability(scored)
    elif page == "System Metrics":
        render_system_metrics(scored)
    else:
        render_simulator()


def render_overview(scored: pd.DataFrame) -> None:
    """Render executive monitoring overview."""
    st.title("Real-Time UPI Transaction Risk Intelligence Platform")
    st.caption("Fraud operations center for payment authorization, risk profiling, and explainability.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Transactions", f"{len(scored):,}", "Synthetic UPI stream")
    with c2:
        metric_card("Avg Risk Score", f"{scored['risk_score'].mean():.1f}", "0-100 authorization scale")
    with c3:
        block_rate = (scored["decision"] == "BLOCK").mean()
        metric_card("Block Rate", f"{block_rate * 100:.1f}%", "High-risk decisions")
    with c4:
        metric_card("Fraud Label Rate", f"{scored['is_fraud'].mean() * 100:.1f}%", "Training target prevalence")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-title">Live Decision Distribution</div>', unsafe_allow_html=True)
        decision_counts = scored["decision"].value_counts().reset_index()
        decision_counts.columns = ["decision", "count"]
        fig = px.bar(
            decision_counts,
            x="decision",
            y="count",
            color="decision",
            color_discrete_map=DECISION_COLORS,
            text_auto=True,
        )
        fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(scored, x="risk_score", nbins=25, color="risk_level", color_discrete_map=RISK_COLORS)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Highest-Risk Transactions</div>', unsafe_allow_html=True)
    st.dataframe(
        scored[
            [
                "transaction_id",
                "user_id",
                "merchant_id",
                "amount",
                "risk_score",
                "risk_level",
                "decision",
                "top_risk_factors",
            ]
        ].head(15),
        use_container_width=True,
        hide_index=True,
    )


def render_transaction_monitor(scored: pd.DataFrame) -> None:
    """Render filterable transaction operations table."""
    st.title("Transaction Monitor")
    c1, c2, c3 = st.columns(3)
    risk_filter = c1.multiselect("Risk level", ["Low", "Medium", "High"], default=["Medium", "High"])
    decision_filter = c2.multiselect("Decision", ["ALLOW", "OTP_VERIFICATION", "BLOCK"], default=["OTP_VERIFICATION", "BLOCK"])
    query = c3.text_input("Search transaction, user, or merchant")

    filtered = scored[scored["risk_level"].isin(risk_filter) & scored["decision"].isin(decision_filter)]
    if query:
        mask = (
            filtered["transaction_id"].str.contains(query, case=False, na=False)
            | filtered["user_id"].str.contains(query, case=False, na=False)
            | filtered["merchant_id"].str.contains(query, case=False, na=False)
        )
        filtered = filtered[mask]

    st.caption(f"{len(filtered):,} transactions match current filters")
    st.dataframe(
        filtered[
            [
                "transaction_id",
                "created_at",
                "user_id",
                "merchant_id",
                "merchant_category",
                "amount",
                "risk_score",
                "risk_level",
                "decision",
                "decision_reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_risk_analytics(scored: pd.DataFrame) -> None:
    """Render fraud, merchant, hourly, and model-performance analytics."""
    st.title("Risk Analytics")
    c1, c2 = st.columns(2)
    with c1:
        fraud_counts = scored["is_fraud"].map({0: "Legitimate", 1: "Fraud"}).value_counts().reset_index()
        fraud_counts.columns = ["label", "count"]
        fig = px.pie(fraud_counts, values="count", names="label", hole=0.5)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hourly = scored.groupby("transaction_hour").agg(avg_risk=("risk_score", "mean"), volume=("transaction_id", "count")).reset_index()
        fig = px.line(hourly, x="transaction_hour", y="avg_risk", markers=True)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        merchant_trends = (
            scored.groupby("merchant_category")
            .agg(avg_risk=("risk_score", "mean"), transactions=("transaction_id", "count"))
            .reset_index()
            .sort_values("avg_risk", ascending=False)
        )
        fig = px.bar(merchant_trends, x="merchant_category", y="avg_risk", color="transactions")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        decision_counts = scored.groupby(["risk_level", "decision"]).size().reset_index(name="count")
        fig = px.bar(decision_counts, x="risk_level", y="count", color="decision", color_discrete_map=DECISION_COLORS)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    render_model_curves()


def render_user_profiles() -> None:
    """Render user profile risk assessment."""
    st.title("User Profiles")
    user_profiles, _ = load_profiles()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("High Risk Users", f"{(user_profiles['user_risk_level'] == 'High').sum():,}")
    with c2:
        metric_card("Median User Risk", f"{user_profiles['user_risk_score'].median():.0f}")
    with c3:
        metric_card("Tracked Users", f"{len(user_profiles):,}")
    fig = px.histogram(user_profiles, x="user_risk_score", color="user_risk_level", color_discrete_map=RISK_COLORS)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(user_profiles.head(100), use_container_width=True, hide_index=True)


def render_merchant_profiles() -> None:
    """Render merchant profile risk assessment."""
    st.title("Merchant Profiles")
    _, merchant_profiles = load_profiles()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("High Risk Merchants", f"{(merchant_profiles['merchant_risk_level'] == 'High').sum():,}")
    with c2:
        metric_card("Avg Fraud Rate", f"{merchant_profiles['merchant_fraud_rate'].mean() * 100:.1f}%")
    with c3:
        metric_card("Tracked Merchants", f"{len(merchant_profiles):,}")
    fig = px.scatter(
        merchant_profiles,
        x="merchant_transaction_volume",
        y="merchant_risk_score",
        color="merchant_category",
        size="merchant_failure_rate",
        hover_data=["merchant_id", "merchant_fraud_rate", "merchant_dispute_rate"],
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(merchant_profiles.head(100), use_container_width=True, hide_index=True)


def render_explainability(scored: pd.DataFrame) -> None:
    """Render SHAP/fallback explanations and feature importance."""
    st.title("Model Explainability")
    sample = scored.iloc[0].to_dict()
    transaction_id = st.selectbox("Transaction", scored["transaction_id"].head(250), index=0)
    selected = scored[scored["transaction_id"] == transaction_id].iloc[0].to_dict() if transaction_id else sample
    explanation = explain_transaction(selected)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Risk Score", str(int(selected["risk_score"])))
    with c2:
        st.markdown(risk_chip(str(selected["risk_level"])), unsafe_allow_html=True)
    with c3:
        st.markdown(decision_chip(str(selected["decision"])), unsafe_allow_html=True)

    contributions = pd.DataFrame(explanation["top_contributions"])
    fig = px.bar(contributions, x="contribution", y="feature", orientation="h", title=f"Waterfall Drivers ({explanation['method']})")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    importance = feature_importance().head(20)
    fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Feature Importance Ranking")
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=45, b=10), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def render_system_metrics(scored: pd.DataFrame) -> None:
    """Render model and platform health metrics."""
    st.title("System Metrics")
    info = model_info()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Model", info["model_name"])
    with c2:
        metric_card("Training Rows", f"{info['training_rows']:,}")
    with c3:
        metric_card("ROC-AUC", f"{info['metrics'].get('roc_auc', 0):.3f}")
    with c4:
        metric_card("PR-AUC", f"{info['metrics'].get('pr_auc', 0):.3f}")

    st.json({"trained_at": info["trained_at"], "feature_count": len(info["features"])})
    st.dataframe(
        scored.groupby("decision")
        .agg(volume=("transaction_id", "count"), avg_risk=("risk_score", "mean"), fraud_rate=("is_fraud", "mean"))
        .reset_index(),
        use_container_width=True,
        hide_index=True,
    )


def render_simulator() -> None:
    """Render centerpiece transaction risk simulator."""
    st.title("Transaction Risk Simulator")
    left, right = st.columns([0.9, 1.1])
    with left:
        amount = st.number_input("Amount", min_value=0.0, value=4500.0, step=100.0)
        hour = st.slider("Transaction Time", 0, 23, 23)
        merchant_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES, index=MERCHANT_CATEGORIES.index("gaming"))
        payment_instrument = st.selectbox("Payment Instrument", PAYMENT_INSTRUMENTS)
        device_changed = st.checkbox("Device Changed", value=True)
        location_changed = st.checkbox("Location Changed", value=True)
        failed_attempts = st.slider("Failed Attempts", 0, 8, 2)
        transactions_last_hour = st.slider("Transaction Frequency", 0, 20, 6)
        merchant_risk = st.slider("Merchant Fraud Rate", 0.0, 1.0, 0.18, 0.01)
        merchant_failure = st.slider("Merchant Failure Rate", 0.0, 1.0, 0.16, 0.01)
        user_velocity = st.slider("User Velocity Score", 0.0, 100.0, 70.0, 1.0)

    payload = {
        "amount": amount,
        "transaction_hour": hour,
        "merchant_category": merchant_category,
        "payment_instrument": payment_instrument,
        "device_changed": device_changed,
        "location_changed": location_changed,
        "failed_attempts": failed_attempts,
        "transactions_last_hour": transactions_last_hour,
        "avg_amount_deviation": max(amount / 1800, 0.1),
        "merchant_fraud_rate": merchant_risk,
        "merchant_failure_rate": merchant_failure,
        "merchant_dispute_rate": merchant_risk / 3,
        "user_velocity_score": user_velocity,
        "device_age_days": 7 if device_changed else 220,
        "account_age_days": 45,
    }
    result = score_transaction(payload)
    explanation = explain_transaction(payload)

    with right:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Risk Score", f"{result['risk_score']}/100")
        with c2:
            st.markdown(risk_chip(result["risk_level"]), unsafe_allow_html=True)
        with c3:
            st.markdown(decision_chip(result["decision"]), unsafe_allow_html=True)
        st.info(result["reason"])
        st.markdown("Top Risk Factors")
        st.write(", ".join(result["top_risk_factors"]) or "No elevated factors detected")

        breakdown = pd.DataFrame(
            [
                {"component": "ML Risk Score", "score": result["risk_score"]},
                {"component": "User Risk Score", "score": result["user_risk_score"]},
                {"component": "Merchant Risk Score", "score": result["merchant_risk_score"]},
            ]
        )
        fig = px.bar(breakdown, x="component", y="score", color="component", range_y=[0, 100])
        fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        contributions = pd.DataFrame(explanation["top_contributions"])
        fig = px.bar(contributions, x="contribution", y="feature", orientation="h", title="SHAP/Fallback Explanation")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def render_model_curves() -> None:
    """Render confusion matrix, ROC, and precision-recall charts."""
    info = model_info()
    metrics = info.get("metrics", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        fig = px.imshow(matrix, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(title="Confusion Matrix", height=320, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        roc = metrics.get("roc_curve", {})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc.get("fpr", []), y=roc.get("tpr", []), mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", height=320, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        pr = metrics.get("precision_recall_curve", {})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr.get("recall", []), y=pr.get("precision", []), mode="lines", name="PR"))
        fig.update_layout(title="Precision-Recall Curve", height=320, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    app()
