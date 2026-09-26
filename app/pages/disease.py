"""Disease Analysis — per-disease model summary + predicted vs actual."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.layout import page_header, html_grid
from components.theme import MUTED, CYAN_DARK
from utils.predict import (
    DISEASES, best_disease_model_r2, shap_global, historical_disease,
)


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    st.markdown("### 🦠 Disease Analysis")
    st.caption(
        "Five Phase-6 disease-specific models. The strongest driver of "
        "each target is shown alongside the winning model's test R²."
    )

    # Summary grid — one card per disease
    best = best_disease_model_r2()
    cards = []
    for d in DISEASES:
        r = best[best["task"] == d["key"]]
        if r.empty:
            r2_str = "—"
            model  = "—"
            rmse   = "—"
        else:
            r2 = float(r["r2"].iloc[0])
            r2_str = f"{r2:.3f}"
            model  = r["model"].iloc[0]
            rmse   = f"{float(r['rmse'].iloc[0]):.2f}"
        # Strong / weak label
        r2_val = float(r["r2"].iloc[0]) if not r.empty else 0.0
        if r2_val >= 0.7: tag, tcol = "Strongly weather-driven", "#10B981"
        elif r2_val >= 0.3: tag, tcol = "Moderately weather-driven", "#F59E0B"
        else: tag, tcol = "Weak / weather is minor factor", "#EF4444"
        cards.append(f"""
<div class="cg-disease" style="height:160px">
  <div class="cg-disease-head">{d['icon']} {d['label']}</div>
  <div>
    <div style="font-size:1.6rem;font-weight:700;color:{MUTED}">R² = {r2_str}</div>
    <div style="font-size:0.78rem;color:{MUTED};margin-top:4px">
      winning model: <b>{model}</b><br>
      test RMSE: <b>{rmse}</b>
    </div>
    <span class="cg-badge" style="background:{tcol}22;color:{tcol};margin-top:8px">{tag}</span>
  </div>
</div>
""")
    st.markdown(html_grid(cards, columns=5), unsafe_allow_html=True)

    st.write("")

    # Per-disease detail with predicted vs actual & top SHAP
    for d in DISEASES:
        st.markdown(f"#### {d['icon']} {d['label']}")
        cols = st.columns([2, 3])

        with cols[0]:
            sub = historical_disease(country["country_code"], d["key"], n_weeks=52)
            if not sub.empty:
                pred_col = f"pred_{d['target']}"
                plot = sub.rename(columns={d["target"]: "Actual", pred_col: "Predicted"})
                series = [c for c in ("Actual", "Predicted") if c in plot.columns]
                fig = px.line(plot, x="date", y=series,
                              color_discrete_sequence=[CYAN_DARK, "#F59E0B"],
                              labels={"value": d["target"], "date": "",
                                      "variable": ""})
                fig.update_layout(
                    height=240, margin=dict(t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color=MUTED),
                    title="Actual vs predicted (last 52 test weeks)",
                    legend=dict(orientation="h", y=-0.2, x=0),
                )
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})

        with cols[1]:
            g = shap_global(d["key"], top_k=8)
            if g is not None:
                top = g.iloc[::-1]
                fig = px.bar(top, x="mean_abs_shap", y="feature",
                             orientation="h",
                             color_discrete_sequence=[CYAN_DARK])
                fig.update_layout(
                    height=240, margin=dict(t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color=MUTED),
                    title=f"Top SHAP features — {d['label']}",
                )
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})

        st.markdown('<div class="cg-footnote">'
                    'Phase 6 winner model on the test split; predictions are '
                    'computed in the app from the saved model.</div>',
                    unsafe_allow_html=True)
        st.write("")
