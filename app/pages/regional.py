"""Regional Analysis — country/region comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.layout import page_header
from components.theme import MUTED
from utils.predict import list_countries, _load_advisories, risk_score_0_100


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    panel = {c["country_code"]: c for c in list_countries()}
    adv = _load_advisories()

    st.markdown("### 🌍 Regional Analysis")
    st.caption("Compare countries across the dataset.")

    # Per-country mean risk score
    grp = adv.groupby("country_code").agg(
        n=("date", "count"),
        pct_high=("predicted_risk_class",
                  lambda s: float((s == "High").mean())),
        mean_p=("p_high", "mean"),
    ).reset_index()
    grp["country"] = grp["country_code"].map(lambda c: panel[c]["country_name"])
    grp["region"] = grp["country_code"].map(lambda c: panel[c]["region"])
    grp["score"]  = grp["mean_p"].apply(risk_score_0_100)
    grp = grp.sort_values("score", ascending=False)

    fig = px.bar(grp, x="score", y="country", color="region",
                 orientation="h", height=520,
                 labels={"score": "Mean risk score (0-100)",
                         "country": ""},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(margin=dict(t=10, b=0, l=120),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=11, color=MUTED),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Region averages
    st.markdown("#### Region averages")
    region = (grp.groupby("region")
                 .agg(n=("country_code", "count"),
                      mean_score=("score", "mean"),
                      pct_high=("pct_high", "mean"))
                 .reset_index()
                 .sort_values("mean_score", ascending=False))
    region["mean_score"] = region["mean_score"].round(1)
    region["pct_high"]   = region["pct_high"].apply(lambda v: f"{v:.1%}")
    st.dataframe(region, width="stretch", hide_index=True)
