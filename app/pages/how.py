"""How It Works — pipeline walkthrough for non-technical users."""

from __future__ import annotations

import streamlit as st

from components.layout import page_header


STEPS = [
    ("📥", "Raw climate + health data",
     "We start with 14,050 weekly rows from 25 countries (2015-01-04 → "
     "2025-10-19). Each row has 30 raw columns — climate, air quality, "
     "disease rates, and country-level metadata."),
    ("🧹", "Cleaning",
     "374 negative AQI readings are clamped to 0. Disease rates and counts "
     "are clipped to ≥ 0. Healthcare access is clipped to ≤ 100. Nothing "
     "else is imputed."),
    ("🛠️", "Feature engineering",
     "About 160 engineered columns are produced per row: rolling means / "
     "sums / maxes at 4 / 8 / 12 weeks, lags at 1 / 2 / 4 / 8 weeks, "
     "cyclical month + week sin/cos encodings, and interactions "
     "(temperature × PM2.5, etc.). All rolling + lag features are built "
     "grouped by country so no future information leaks across borders."),
    ("🎯", "Feature selection",
     "Near-duplicate columns (|r| ≥ 0.95) are removed, then three rankings — "
     "Pearson correlation, mutual information, and tree importance — vote on "
     "which features to keep (capped at 60). 54 features survive the vote, and "
     "the models train on 50 of them."),
    ("🤖", "Modelling",
     "A chronological split is used: train ≤ 2022, val = 2023, test ≥ 2024. "
     "We train one overall classifier (Low / Medium / High) and five disease "
     "regressors (respiratory, vector-borne, heat-related, waterborne, "
     "cardiovascular). All six share the same 50 input features; each is "
     "tuned separately and the winner is picked on the validation year."),
    ("📋", "Advisory engine",
     "Rule-based: if respiratory risk is elevated → recommend mask / reduced "
     "exposure; if vector risk is high → mosquito nets / repellents; if heat "
     "risk is up → hydration / shade; etc. The advisory never claims to be "
     "individual medical advice."),
    ("🔍", "SHAP explanations",
     "For every model we compute SHAP feature attributions on the test split. "
     "This shows what pushed a particular prediction up or down — so you can "
     "see exactly why a region is flagged as High."),
    ("🌐", "Dashboard",
     "All of the above becomes the dashboard you see now. Every number, "
     "score, and feature importance is pulled live from the pipeline "
     "artefacts — there is no fake data."),
]


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    st.markdown("### ❓ How It Works")
    st.caption("The pipeline in eight steps.")

    for i, (icon, title, body) in enumerate(STEPS, start=1):
        st.markdown(f"""
<div class="cg-card" style="margin-bottom:10px">
  <div style="display:flex;gap:14px;align-items:flex-start">
    <div style="font-size:1.6rem;line-height:1">{icon}</div>
    <div>
      <div style="font-weight:600;font-size:1.0rem">{i}. {title}</div>
      <div class="cg-footnote" style="margin-top:4px;font-style:normal;color:{ "#334155"}">{body}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
