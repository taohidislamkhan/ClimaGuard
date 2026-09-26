"""My Risk page — country/date picker + overall & per-disease risks."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from components.layout import page_header, disease_card, html_grid
from components.theme import MUTED, CYAN_DARK, RISK_BG, RISK_FG
from utils.predict import (
    DISEASES, assess, risk_score_0_100,
    historical_overall, historical_disease,
)


DISCLAIMER = (
    "Regional environmental risk estimate — not a medical diagnosis for any "
    "individual."
)


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    a = assess(country["country_code"], date)
    if a is None:
        st.error("No data for this selection."); return

    # Overall risk hero (smaller than dashboard)
    score = risk_score_0_100(a.p_high)
    cls = a.predicted_risk_class
    color = RISK_FG[cls]
    bg = RISK_BG[cls]

    st.markdown(f"""
<div class="cg-hero">
  <div class="cg-hero-score">
    <div class="cg-hero-num">
      <div class="big">{score:.0f}</div>
      <div class="sub">/ 100 · P(High) = {a.p_high:.2f}</div>
    </div>
  </div>
  <div class="cg-hero-body">
    <h2>Overall Environmental Risk — {country['country_name']}</h2>
    <div class="cg-hero-tag">
      <span class="cg-badge cg-badge-{cls}" style="background:{bg};color:{color}">
        {cls.upper()} RISK
      </span>
    </div>
    <div class="cg-hero-meta">
      {a.advisory.split('|', 1)[-1].strip() if '|' in a.advisory else a.advisory}
      <br><span class="cg-footnote">{DISCLAIMER}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.write("")

    # Disease cards
    st.markdown("### Disease-specific risks")
    cards = []
    for d in DISEASES:
        pct = a.disease_pctile.get(d["key"])
        d_score = None if pct is None else pct * 100
        d_level = a.disease_level.get(d["key"])
        sub = historical_disease(country["country_code"], d["key"], n_weeks=12)
        spark = sub[d["target"]].tail(8).tolist() if not sub.empty else []
        cards.append(disease_card(d["key"], d_level, d_score,
                                  delta_text=None, spark=spark))
    st.markdown(html_grid(cards, columns=5), unsafe_allow_html=True)
    st.markdown('<div class="cg-footnote">Score = percentile of the model\'s '
                'predicted value within this country\'s 2015-2025 history.</div>',
                unsafe_allow_html=True)

    st.write("")

    # Historical trend
    st.markdown("### P(High) over the last 12 weeks")
    hist = historical_overall(country["country_code"], n_weeks=12)
    if not hist.empty:
        hist["score"] = hist["p_high"].apply(risk_score_0_100)
        fig = px.line(hist, x="date", y="score", markers=True,
                      color_discrete_sequence=[CYAN_DARK])
        fig.update_layout(
            height=320, margin=dict(t=10, b=0),
            yaxis=dict(title="Risk score (0-100)", range=[0, 100]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color=MUTED),
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(f'<div class="cg-footnote">{DISCLAIMER}</div>',
                unsafe_allow_html=True)
