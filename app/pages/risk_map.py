"""Risk Map — full-screen Plotly choropleth with risk-switcher + date scrubber."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.layout import page_header
from components.theme import MUTED
from utils.predict import (
    DISEASES, DISEASE_BY_KEY, regional_snapshot, _load_advisories,
)


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    st.markdown("### 🗺️ Regional Environmental Risk Map")
    st.caption(
        "25 countries · 8 regions · weekly · test split 2024-01-07 → 2025-10-19. "
        "Hover for country-level detail."
    )

    ctrl = st.columns([2, 2, 2])
    with ctrl[0]:
        opts = [("overall_classifier", "Overall")] + [
            (d["key"], d["label"]) for d in DISEASES
        ]
        labels = [o[1] for o in opts]
        keys   = [o[0] for o in opts]
        idx = st.selectbox("Risk to display", labels,
                           index=0, key="map_metric")
        metric_key = keys[labels.index(idx)]
    with ctrl[1]:
        level = st.radio("Colour scale", ["Class (Low/Med/High)", "Mean score"],
                         horizontal=True, key="map_level")
    with ctrl[2]:
        adv_dates = _load_advisories()["date"]
        mn, mx = adv_dates.min().date(), adv_dates.max().date()
        d = st.date_input("As of week", value=date.date(),
                          min_value=mn, max_value=mx, key="map_date")
        date = pd.Timestamp(d)

    is_disease = metric_key != "overall_classifier"
    if level.startswith("Class"):
        snap = regional_snapshot(date, level="class", task=metric_key)
        if snap.empty:
            st.warning("No data in this window."); return
        if is_disease:
            # value = percentile of predicted value vs the country's history
            snap["level"] = snap["value"].apply(
                lambda v: "High" if v >= 0.66 else ("Medium" if v >= 0.33 else "Low"))
            value_lab = "Percentile vs own history"
        else:
            # value = share of weeks predicted High
            snap["level"] = snap["value"].apply(
                lambda v: "High" if v >= 0.5 else ("Medium" if v >= 0.2 else "Low"))
            value_lab = "% High weeks"
        fig = px.choropleth(
            snap, locations="country_code", locationmode="ISO-3",
            color="level",
            color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
            category_orders={"level": ["Low", "Medium", "High"]},
            hover_name="country_name",
            hover_data={"country_code": False, "value": ":.0%"},
            labels={"level": "Risk band", "value": value_lab},
        )
    else:
        snap = regional_snapshot(date, level="score", task=metric_key)
        if snap.empty:
            st.warning("No data in this window."); return
        if is_disease:
            d = DISEASE_BY_KEY[metric_key]
            clab, fmt, rng = f"Mean predicted {d['target']}", ":.2f", None
        else:
            clab, fmt, rng = "Mean P(High)", ":.2f", (0, 1)
        fig = px.choropleth(
            snap, locations="country_code", locationmode="ISO-3",
            color="value", color_continuous_scale="RdYlGn_r",
            range_color=rng,
            hover_name="country_name",
            hover_data={"country_code": False, "value": fmt},
            labels={"value": clab},
        )
    fig.update_layout(
        height=560, margin=dict(t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(showframe=False, showcoastlines=True,
                 projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter, sans-serif", size=11, color=MUTED),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Region summary
    st.markdown("#### Per-region summary")
    if not snap.empty and "level" in snap.columns:
        rs = (snap.groupby("region")
                .agg(n=("country_code", "count"),
                     high=("level", lambda s: int((s == "High").sum())))
                .reset_index())
        rs["pct_high"] = (rs["high"] / rs["n"]).fillna(0).apply(lambda v: f"{v:.0%}")
        st.dataframe(rs, width="stretch", hide_index=True)
    elif not snap.empty:
        rs = (snap.groupby("region")
                .agg(n=("country_code", "count"), mean_value=("value", "mean"))
                .reset_index())
        rs["mean_value"] = rs["mean_value"].round(2)
        st.dataframe(rs, width="stretch", hide_index=True)
    st.markdown('<div class="cg-footnote">Regional risk estimate, not '
                'an individual medical diagnosis.</div>',
                unsafe_allow_html=True)
