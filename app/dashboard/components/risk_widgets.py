"""Reusable Streamlit widgets for risk operations views."""

from __future__ import annotations

import streamlit as st

from app.config import DECISION_COLORS, RISK_COLORS


def inject_global_styles() -> None:
    """Apply compact fintech dashboard styling."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1420px;}
        [data-testid="stSidebar"] {background: #0f172a;}
        [data-testid="stSidebar"] * {color: #f8fafc;}
        h1, h2, h3 {letter-spacing: 0;}
        .metric-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .metric-label {font-size: 0.78rem; color: #64748b; margin-bottom: 6px;}
        .metric-value {font-size: 1.75rem; font-weight: 700; color: #0f172a;}
        .metric-delta {font-size: 0.82rem; color: #475569; margin-top: 4px;}
        .chip {
            border-radius: 999px;
            padding: 4px 10px;
            color: white;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
        }
        .section-title {font-size: 1.05rem; font-weight: 700; color: #0f172a; margin: 16px 0 8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "") -> None:
    """Render a compact metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_chip(label: str) -> str:
    """Return HTML for a risk category chip."""
    color = RISK_COLORS.get(label, "#64748b")
    return f'<span class="chip" style="background:{color};">{label}</span>'


def decision_chip(label: str) -> str:
    """Return HTML for a decision chip."""
    color = DECISION_COLORS.get(label, "#64748b")
    text = label.replace("_", " ")
    return f'<span class="chip" style="background:{color};">{text}</span>'

