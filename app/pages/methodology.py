"""Data & Methodology — dataset card, correlation heatmap, limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.layout import page_header
from components.data import load_meta, CHARTS_DIR


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    st.markdown("### 📊 Data & Methodology")

    meta = load_meta()
    eng = meta.get("engineering", {})
    sel = meta.get("selection", {})

    # ----- Dataset card
    st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-head">📦 Dataset</div>
  <div class="cg-env-grid" style="grid-template-columns:repeat(4,1fr);gap:12px">
    <div class="cg-env-cell">
      <div class="lbl">Source</div>
      <div class="val" style="font-size:0.92rem">global_climate_health_impact_tracker_2015_2025.csv</div>
    </div>
    <div class="cg-env-cell">
      <div class="lbl">Rows (post-clean)</div>
      <div class="val">{eng.get('rows', '—'):,}</div>
      <div class="sub">0 nulls · 0 duplicate rows</div>
    </div>
    <div class="cg-env-cell">
      <div class="lbl">Engineered predictors</div>
      <div class="val">{eng.get('n_predictors', '—')}</div>
      <div class="sub">30 raw cols</div>
    </div>
    <div class="cg-env-cell">
      <div class="lbl">Selection committee</div>
      <div class="val">{sel.get('committee_size', '—')}</div>
      <div class="sub">top-k after dedup + vote</div>
    </div>
    <div class="cg-env-cell">
      <div class="lbl">Countries</div>
      <div class="val">25</div>
    </div>
    <div class="cg-env-cell">
      <div class="lbl">Regions</div>
      <div class="val">8</div>
    </div>
    <div class="cg-env-cell">
      <div class="lbl">Time range</div>
      <div class="val" style="font-size:0.95rem">2015-01-04 → 2025-10-19</div>
    </div>
    <div class="cg-env-cell">
      <div class="lbl">Test split</div>
      <div class="val">2024+</div>
      <div class="sub">chronological</div>
    </div>
  </div>
  <div class="cg-footnote" style="margin-top:10px">
    Cleaning actions (from Phase 1): 374 negative AQI values clamped to 0;
    disease rates/counts clipped to ≥ 0; healthcare access clipped to ≤ 100.
    Nothing else was imputed or fabricated.
  </div>
</div>
""", unsafe_allow_html=True)

    st.write("")

    # ----- Correlation heatmap
    heatmap = CHARTS_DIR / "correlation_heatmap.png"
    if heatmap.exists():
        st.markdown("#### Correlation structure")
        st.image(str(heatmap), caption="Pearson correlation matrix — "
                 "climate, air-quality, and disease-rate columns.",
                 width="stretch")
        st.markdown("""
| Pair | Pearson r |
|---|---|
| PM2.5 ↔ AQI | ≈ 0.97 |
| PM2.5 ↔ respiratory rate | ≈ 0.76 |
| Heat-wave days ↔ heat admissions | ≈ 0.70 |
| Temperature ↔ vector risk | ≈ 0.65 |
""")

    # ----- Feature engineering summary
    st.markdown("#### Feature engineering summary")
    rolling = eng.get("rolling_windows", [])
    lags = eng.get("lag_weeks", [])
    composite = eng.get("composite_cols", [])
    rows = [
        {"Family": "Rolling windows",
         "Spec":   f"mean / max / sum at {', '.join(map(str, rolling))} weeks"},
        {"Family": "Lag features",
         "Spec":   f"weeks {', '.join(map(str, lags))}"},
        {"Family": "Cyclical encodings",
         "Spec":   "month_sin/cos, week_sin/cos"},
        {"Family": "Interaction terms",
         "Spec":   "temp×PM2.5, rain×temp, PM2.5×AQI"},
        {"Family": "Grouped country aggregates",
         "Spec":   "rolling/lag features built per-country (no leakage)"},
        {"Family": "Categorical dummies",
         "Spec":   "region, climate_zone, income_level, hemisphere"},
        {"Family": "Composite disease targets",
         "Spec":   ", ".join(composite) if composite else "—",
         "Note":   "target-only, never used as predictors"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ----- Methodology notes
    st.markdown("#### Evaluation methodology")
    st.markdown(
        f"""
**Chronological split.** Training ≤ 2022, validation = 2023, test ≥ 2024.
A random split would leak future weeks into training — disease rates
in the engineered lag/rolling features would not be "future" at decision
time. The chronological split is the right way to evaluate a forecasting
model even when the model isn't a pure time-series.

**Feature selection.** Pearson dedup @ |r|≥0.95 → three rankings
(Pearson / MI / tree importance) → keep features with ≥ 2 of 3 votes,
top-60 cap ({sel.get('committee_size', '—')} survived the vote).

**Determinism.** Every script pins `random_state=42`; step5/6/7 also
seed Python, NumPy, and framework RNGs. Two back-to-back runs produce
identical artefacts (SHA-256 verified).
        """
    )

    # ----- Limitations (prominent, not buried)
    st.markdown("""
<div class="cg-card" style="border:1px solid #FCA5A5;background:#FEF2F2">
  <div class="cg-card-head" style="color:#991B1B">⚠️ Limitations</div>
  <ul style="margin:0;padding-left:20px;color:#7F1D1D;line-height:1.7">
    <li>Prototype for academic / research use — not a clinical diagnostic tool.</li>
    <li>Predicts <b>regional environmental risk</b>, not individual disease.</li>
    <li>Dataset does <b>not</b> contain humidity, wind speed, UV index, elevation,
        BMI, smoking, asthma history, or named per-disease case counts.</li>
    <li>Historical dataset is <b>weekly</b> — daily granularity is not modelled.</li>
    <li>Model performance varies substantially by disease target.</li>
    <li><b>Cardiovascular model has low / negative R²</b> — climate is a weak
        predictor; other drivers dominate. Reported honestly as a negative result.</li>
    <li>Predictions should <b>not</b> be interpreted as medical advice.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
