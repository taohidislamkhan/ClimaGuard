"""Dashboard page — the landing screen, matching the reference screenshot."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from components.layout import (
    page_header, disease_card, shap_bars, env_cell, regional_profile_card,
    html_grid,
)
from components.theme import (
    CYAN_DARK, INK, MUTED, RISK_BG, RISK_FG, HERO_ILLU_SVG, render_ring_score,
)
from utils.predict import (
    DISEASES, assess, risk_score_0_100, level_for_p_high,
    historical_overall, historical_disease, what_changed, regional_snapshot,
    nearest_row, shap_local, shap_global, _load_clean,
)


DISCLAIMER = (
    "Regional environmental risk estimate for the selected country and "
    "week — not a medical diagnosis for any individual."
)


def _what_changed_block(changes: dict | None) -> str:
    """Render the 'What changed today?' card content."""
    if not changes:
        return "<div class='cg-footnote'>No prior week within 4 weeks — change unavailable.</div>"
    weeks_back = changes.get("weeks_back", 0)
    fields = [
        ("pm25_ugm3",                "PM2.5",                    "µg/m³"),
        ("temperature_celsius",      "Temperature",              "°C"),
        ("temp_anomaly_celsius",     "Temp anomaly",             "°C"),
        ("heat_wave_days",           "Heat-wave days",           "days"),
        ("extreme_weather_events",   "Extreme weather events",   ""),
        ("precipitation_mm",         "Precipitation",            "mm"),
        ("p_high",                   "Predicted P(High)",        ""),
    ]
    rows = []
    for key, label, unit in fields:
        if key not in changes:
            continue
        v = changes[key]
        abs_d = v["abs"]
        pct = v["pct"]
        arrow = "↑" if abs_d > 0 else ("↓" if abs_d < 0 else "·")
        color = "#EF4444" if abs_d > 0 else ("#10B981" if abs_d < 0 else MUTED)
        if "pts" in v:
            pct_str = f"{v['pts']:+.1f} pts"
        else:
            pct_str = f"{pct:+.1f}%" if pct is not None else "—"
        rows.append(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:6px 0;border-bottom:1px dashed #E2E8F0">
  <span style="color:{INK};font-weight:500">{label}</span>
  <span style="color:{color};font-weight:600">{arrow} {pct_str}</span>
</div>""")
    return "\n".join(rows)


def render(state: dict) -> None:
    """Render the full Dashboard page."""
    country = state["country"]
    date = state["date"]

    # 1. Page header
    page_header(country["country_name"], country["region"], date)

    # 2. Inference
    a = assess(country["country_code"], date)
    if a is None:
        st.error("No model output available for this selection.")
        return

    score = risk_score_0_100(a.p_high)
    # The badge shows the classifier's own class, matching the rule engine.
    level_disp = a.predicted_risk_class
    color = RISK_FG[level_disp]
    bg = RISK_BG[level_disp]

    # 3. Hero card — left: ring score; right: title + meta + CTA
    ring = render_ring_score(score, level_disp, size=200)
    hero = f"""
<div class="cg-hero">
  <div class="cg-hero-score">
    {ring}
    <div class="cg-hero-num">
      <div class="big">{score:.0f}</div>
      <div class="sub">/ 100</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 220px;gap:24px">
    <div class="cg-hero-body">
      <h2>Your Regional Environmental Risk</h2>
      <div class="cg-hero-tag">
        <span class="cg-badge cg-badge-{level_disp}" style="background:{bg};color:{color}">
          {level_disp.upper()} RISK
        </span>
      </div>
      <div class="cg-hero-meta">
        Based on current environmental and regional conditions.
        <br><span class="cg-footnote">{DISCLAIMER}</span>
      </div>
    </div>
    <div class="cg-illu">
      <div class="sun"></div>
      {HERO_ILLU_SVG}
    </div>
  </div>
</div>
"""
    st.markdown(hero, unsafe_allow_html=True)

    # CTA buttons in a row beneath the hero
    cta_l, cta_r = st.columns([1, 5])
    with cta_l:
        st.markdown('<div class="cg-cta">', unsafe_allow_html=True)
        if st.button("View Detailed Analysis →", key="cta_detail"):
            st.session_state["page"] = "explain"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with cta_r:
        st.markdown(
            '<div class="cg-recalc-btn">', unsafe_allow_html=True)
        if st.button("↻ Recalculate Risk", key="cta_recalc"):
            st.cache_data.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # 4. Disease cards row
    hist = historical_overall(country["country_code"], n_weeks=12)
    cards = []
    for d in DISEASES:
        # Per-disease percentile of the predicted value → score 0-100
        pct = a.disease_pctile.get(d["key"])
        d_score = None if pct is None else pct * 100
        d_level = a.disease_level.get(d["key"])
        # Sparkline: the per-country actual values for this disease
        sub = historical_disease(country["country_code"], d["key"], n_weeks=12)
        spark = sub[d["target"]].tail(8).tolist() if not sub.empty else []
        # Delta vs previous week
        delta_text = ""
        if len(sub) >= 2:
            cur, prv = float(sub[d["target"]].iloc[-1]), float(sub[d["target"]].iloc[-2])
            if prv:
                p = 100 * (cur - prv) / abs(prv)
                sign = "+" if p >= 0 else ""
                delta_text = f"{sign}{p:.1f}% vs prev"
        cards.append(disease_card(d["key"], d_level, d_score, delta_text, spark))
    st.markdown(html_grid(cards, columns=5), unsafe_allow_html=True)

    st.write("")

    # 5. Three-column block: Why elevated · Current environment · Trend
    col1, col2, col3 = st.columns([5, 4, 5])

    # ----- Why is your risk elevated? -----
    with col1:
        # Try local SHAP first
        row = nearest_row(country["country_code"], date)
        local = shap_local("overall_classifier", row)

        rows_for_bars: list[dict] = []
        if local is not None and not local.empty:
            for _, r in local.head(8).iterrows():
                rows_for_bars.append({
                    "label": r["feature"],
                    "magnitude": float(r["abs"]),
                    "contribution_text": f"{float(r['shap_value']):+.2f}",
                })
        else:
            # Fall back to global importance
            g = shap_global("overall_classifier", top_k=8)
            if g is not None:
                for _, r in g.iterrows():
                    rows_for_bars.append({
                        "label": r["feature"],
                        "magnitude": float(r["mean_abs_shap"]),
                        "contribution_text": "—",
                    })

        st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-head">❓ Why is your risk elevated?</div>
  <div class="cg-card-sub">Top environmental and regional factors influencing today's estimate.</div>
  {shap_bars(rows_for_bars)}
  <div class="cg-footnote" style="margin-top:10px">
    {"Local SHAP for this week on the Phase 6 classifier: bars show magnitude, "
     "values are signed contributions to the High-class log-odds."
     if local is not None and not local.empty else
     "Local SHAP unavailable — showing global mean |SHAP| of the Phase 6 classifier."}
  </div>
</div>
""", unsafe_allow_html=True)

    # ----- Current environment -----
    with col2:
        # Read environment fields from the cleaned source CSV, since
        # step6's advisories.csv only stores engineered features.
        # Snapped to the nearest week — same approach as nearest_row().
        clean = _load_clean()
        sub_env = clean[clean["country_code"] == country["country_code"]]
        env_row = None
        if not sub_env.empty:
            pos = (pd.to_datetime(sub_env["date"]) - date).abs().argsort()[:1]
            env_row = sub_env.iloc[pos].iloc[0]
        if env_row is not None:
            temp     = float(env_row.get("temperature_celsius", float("nan")))
            anomaly  = float(env_row.get("temp_anomaly_celsius", float("nan")))
            pm25     = float(env_row.get("pm25_ugm3", float("nan")))
            aqi      = float(env_row.get("air_quality_index", float("nan")))
            precip   = float(env_row.get("precipitation_mm", float("nan")))
            heat     = int(env_row.get("heat_wave_days", 0))
            extreme  = int(env_row.get("extreme_weather_events", 0))
            health   = float(env_row.get("healthcare_access_index", float("nan")))
            cells = [
                env_cell("🌡️", "Temperature",      f"{temp:.1f} °C",    "current"),
                env_cell("🌡️", "Temp anomaly",     f"{anomaly:+.2f} °C", "vs climatology"),
                env_cell("🌫️", "PM2.5",            f"{pm25:.1f} µg/m³", "weekly avg"),
                env_cell("🌀", "AQI",              f"{aqi:.0f}",        "air quality"),
                env_cell("🌧️", "Precipitation",    f"{precip:.1f} mm",  "weekly"),
                env_cell("☀️", "Heat-wave days",   f"{heat}",            "in week"),
                env_cell("⚡", "Extreme weather",  f"{extreme}",         "events"),
                env_cell("🏥", "Healthcare",      f"{health:.0f}",      "access index"),
            ]
        else:
            cells = []
        st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-head">🌦️ Current Environment</div>
  <div class="cg-card-sub">📍 {country['country_name']} · {date.strftime('%b %d, %Y')}</div>
  <div class="cg-env-grid" style="grid-template-columns:repeat(2,1fr);gap:10px;margin-top:6px">
    {''.join(cells)}
  </div>
</div>
""", unsafe_allow_html=True)

    # ----- Weekly trend -----
    with col3:
        sub = hist.tail(8).copy()
        if not sub.empty:
            sub["score"] = sub["p_high"].apply(risk_score_0_100)
            fig = px.line(
                sub, x="date", y="score",
                markers=True, line_shape="spline",
                color_discrete_sequence=[CYAN_DARK],
            )
            fig.update_layout(
                height=200, margin=dict(t=10, b=0, l=0, r=0),
                yaxis=dict(title=None, range=[0, 100],
                           tickvals=[0, 25, 50, 75, 100]),
                xaxis=dict(title=None),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=11, color=MUTED),
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=6))
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
        st.markdown(
            f'<div class="cg-footnote">Recent weekly risk trend for {country["country_name"]}. '
            f'Data is weekly, not daily — showing the last {len(sub)} weeks.</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # 6. Advisories + What changed + Mini map
    col1, col2, col3 = st.columns([5, 3, 4])

    with col1:
        # Render the rule-engine advisory as a coloured callout
        cls_lvl = level_for_p_high(a.p_high)
        st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-head">📋 Today's Advisories</div>
  <div class="cg-callout cg-callout-{cls_lvl.lower()}">
    <b>{a.advisory.split('|')[0].strip() if '|' in a.advisory else 'Advisory'}</b>
    <div style="margin-top:6px">{a.advisory.split('|', 1)[-1].strip() if '|' in a.advisory else a.advisory}</div>
  </div>
  <div class="cg-footnote" style="margin-top:10px">
    Rule-engine advisory from Phase 6. Based on the regional prediction
    for the selected week; not medical advice.
  </div>
</div>
""", unsafe_allow_html=True)

    with col2:
        changes = what_changed(country["country_code"], date)
        rows = _what_changed_block(changes)
        delta_score = ""
        if changes and "p_high" in changes:
            d = changes["p_high"]["abs"]
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
            color = "#EF4444" if d > 0 else ("#10B981" if d < 0 else MUTED)
            delta_score = (
                f'<div style="font-size:1.3rem;font-weight:700;color:{INK};margin:6px 0">'
                f'P(High) {arrow} {abs(d)*100:.0f} pts</div>'
                f'<div style="color:{color};font-weight:600">vs previous week</div>'
            )
        st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-head">⚡ What Changed Today?</div>
  <div class="cg-card-sub">Change from the previous available week.</div>
  {delta_score}
  <div style="margin-top:10px">{rows}</div>
</div>
""", unsafe_allow_html=True)

    with col3:
        snap = regional_snapshot(date, level="class")
        if not snap.empty:
            # Map value (% High) → colour buckets
            def lvl(v):
                if v >= 0.50: return "High"
                if v >= 0.20: return "Medium"
                return "Low"
            snap["level"] = snap["value"].apply(lvl)
            color_map = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
            fig = px.scatter_geo(
                snap, locations="country_code", locationmode="ISO-3",
                lat="latitude", lon="longitude",
                color="level",
                color_discrete_map=color_map,
                hover_name="country_name",
                hover_data={"country_code": False, "latitude": False,
                            "longitude": False, "value": ":.0%"},
                labels={"level": "Risk", "value": "% High weeks"},
                size=snap["value"].apply(lambda v: 6 + v * 18),
            )
            fig.update_layout(
                height=240, margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                geo=dict(showframe=False, showcoastlines=True,
                         projection_type="natural earth",
                         bgcolor="rgba(0,0,0,0)"),
                font=dict(family="Inter, sans-serif", size=10, color=MUTED),
                legend=dict(orientation="h", y=-0.1, x=0, font=dict(size=10)),
            )
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
        st.markdown(
            f'<div class="cg-footnote">Environmental Risk Map — '
            f'{date.strftime("%b %d, %Y")} ± 2 weeks. Bubble size and colour = '
            f'% of weeks in the window predicted High.</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # 7. Regional profile card (the "My Health" → "Region" replacement)
    n_test_rows = len(hist)
    st.markdown(regional_profile_card(country, date, n_test_rows),
                unsafe_allow_html=True)

    # Footer disclaimer
    st.markdown(
        f'<div class="cg-footnote" style="margin-top:14px">{DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
