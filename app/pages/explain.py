"""Explainability — global SHAP, beeswarm, local waterfall for the current query."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.layout import page_header
from components.theme import MUTED, CYAN_DARK
from components.data import SHAP_PNG_DIR
from utils.predict import (
    DISEASES, shap_global, shap_local, nearest_row, load_model, MODEL_FILES,
)


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    st.markdown("### 🔍 Explainability (SHAP)")
    st.caption(
        "Global feature importance per model + local SHAP for the current "
        "query. Red = pushes prediction up, blue = down."
    )

    task = st.selectbox(
        "Model",
        options=[("overall_classifier", "Overall classifier")] + [
            (d["key"], d["label"]) for d in DISEASES
        ],
        index=0, format_func=lambda o: o[1],
    )[0]

    tab_g, tab_b, tab_l = st.tabs(["Global", "Beeswarm (direction)", "Local (this query)"])

    with tab_g:
        g = shap_global(task, top_k=20)
        if g is not None:
            top = g.iloc[::-1]
            fig = px.bar(top, x="mean_abs_shap", y="feature",
                         orientation="h",
                         color_discrete_sequence=[CYAN_DARK])
            fig.update_layout(
                height=520, margin=dict(t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=11, color=MUTED),
            )
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
        else:
            st.warning("No global SHAP table for this model.")

    with tab_b:
        png = SHAP_PNG_DIR / f"shap_summary_{task}.png"
        if png.exists():
            st.image(str(png), caption=f"Beeswarm — {task}",
                     width="stretch")
        else:
            st.info("No beeswarm PNG on disk for this model.")

    with tab_l:
        row = nearest_row(country["country_code"], date)
        if load_model(task) is None:
            st.warning(f"Model file `{MODEL_FILES[task]}` not available.")
        elif row is None:
            st.info("No test-split row for this country/date.")
        else:
            local = shap_local(task, row)
            if local is None or local.empty:
                st.info("Local SHAP could not be computed for this query.")
            else:
                top = local.head(12).iloc[::-1]
                fig = px.bar(
                    top, x="shap_value", y="feature", orientation="h",
                    color=top["shap_value"].apply(
                        lambda v: "#EF4444" if v > 0 else "#0EA5E9"),
                    color_discrete_map="identity",
                )
                fig.update_layout(
                    height=480, margin=dict(t=50, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=11, color=MUTED),
                    title=f"Local SHAP — feature contributions for {country['country_name']} · "
                          f"{date.strftime('%b %d, %Y')}",
                )
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
                if len(local):
                    r1 = local.iloc[0]
                    direction = ("pushed the prediction **up**"
                                 if r1["shap_value"] > 0
                                 else "pushed the prediction **down**")
                    st.markdown(
                        f'<div class="cg-footnote">Top driver for this row: '
                        f'<b>{r1["feature"]}</b> (value={float(r1["value"]):.2f}, '
                        f'SHAP={float(r1["shap_value"]):+.3f}) — {direction}.</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    '<div class="cg-footnote">'
                    + ("Contributions to the High-class log-odds."
                       if task == "overall_classifier" else
                       "Contributions on the target's own scale, relative to "
                       "the model's average prediction.")
                    + '</div>', unsafe_allow_html=True)
