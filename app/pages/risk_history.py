"""Risk History — multi-country historical risk trends."""

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

    adv = _load_advisories()
    panel = {c["country_code"]: c for c in list_countries()}

    st.markdown("### 📈 Risk History")
    st.caption(
        "Compare weekly P(High) trajectories across countries."
    )

    countries = sorted(panel.keys(),
                       key=lambda c: panel[c]["country_name"])
    default = [country["country_code"]]
    if "BRA" in countries and "BRA" != country["country_code"]:
        default.append("BRA")
    if "USA" in countries and "USA" not in default:
        default.append("USA")

    sel = st.multiselect(
        "Countries", options=countries, default=default,
        format_func=lambda c: panel[c]["country_name"],
    )

    if not sel:
        st.info("Select at least one country."); return

    sub = adv[adv["country_code"].isin(sel)].copy()
    sub["score"] = sub["p_high"].apply(risk_score_0_100)
    sub["country"] = sub["country_code"].map(lambda c: panel[c]["country_name"])

    fig = px.line(sub, x="date", y="score", color="country",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(
        height=440, margin=dict(t=10, b=0),
        yaxis=dict(title="Risk score (0-100)", range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color=MUTED),
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="cg-footnote">Weekly P(High) from the Phase 6 '
                'classifier, mapped to a 0-100 score.</div>',
                unsafe_allow_html=True)
