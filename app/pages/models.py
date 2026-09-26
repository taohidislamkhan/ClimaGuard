"""Model Performance — classifier + per-disease R² summary tables."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.layout import page_header
from components.theme import MUTED, CYAN_DARK
from utils.predict import (
    classifier_metrics_test, disease_metrics_test, best_disease_model_r2,
)


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    st.markdown("### 🤖 Model Performance")
    st.caption(
        "Measured metrics from `output/phase6_metrics.csv` (test split, "
        "2024-01-07 → 2025-10-19). Models are not ranked — all measured "
        "metrics are shown for transparency."
    )

    # ----- Overall classifier
    st.markdown("#### Overall environmental risk classifier")
    clf = classifier_metrics_test().copy()
    for c in ("accuracy", "macro_f1", "roc_auc_ovr_weighted"):
        if c in clf.columns:
            clf[c] = clf[c].round(3)
    clf = clf.rename(columns={"roc_auc_ovr_weighted": "roc_auc"})
    show = clf[["model", "accuracy", "macro_f1", "roc_auc", "n_rows"]]
    st.dataframe(show, width="stretch", hide_index=True)

    import plotly.express as px
    melted = clf.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1", "roc_auc"],
        var_name="metric", value_name="value",
    )
    fig = px.bar(melted, x="model", y="value", color="metric",
                 barmode="group",
                 color_discrete_sequence=[CYAN_DARK, "#F59E0B", "#10B981"],
                 title="Classifier baselines — test split (higher = better)")
    fig.update_layout(
        height=360, margin=dict(t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color=MUTED),
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(
        '<div class="cg-footnote">Note: the proposal numbers in the brief '
        '(RF acc 0.617 / F1 0.615 / ROC 0.799) differ from these measured '
        'values. The dashboard reflects what the actual pipeline produced '
        'on the held-out test split.</div>',
        unsafe_allow_html=True,
    )

    # ----- Per-disease R² winners
    st.markdown("#### Disease-specific regressors — winning-model R²")
    best = best_disease_model_r2().copy()
    best["r2"]   = best["r2"].round(3)
    best["rmse"] = best["rmse"].round(2)
    best["mae"]  = best["mae"].round(2)
    best["model"] = best["model"].str.upper()
    best = best.rename(columns={"model": "winner"})
    best["disease"] = best["task"].map({"respiratory": "Respiratory",
                                         "vector": "Vector-borne",
                                         "heat": "Heat-related",
                                         "waterborne": "Waterborne",
                                         "cardio": "Cardiovascular"})
    best = best[["disease", "winner", "r2", "rmse", "mae"]]
    st.dataframe(best, width="stretch", hide_index=True)

    # ----- All test metrics across the 5 disease targets
    st.markdown("#### All Phase-6 models × disease targets (test split)")
    full = disease_metrics_test().copy()
    full["r2"] = full["r2"].round(3)
    full["rmse"] = full["rmse"].round(2)
    full["mae"]  = full["mae"].round(2)
    full = full[["task", "model", "rmse", "mae", "r2", "n_rows"]]
    st.dataframe(full, width="stretch", hide_index=True)
