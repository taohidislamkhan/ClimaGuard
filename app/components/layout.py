"""Reusable layout components: page header, disease cards, SHAP bars, etc."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from components.theme import (
    NAVY, RISK_BG, RISK_FG, BG_CARD, BORDER, MUTED, INK, INK_SOFT,
    CYAN, CYAN_DARK, HERO_ILLU_SVG, risk_color, render_ring_score,
)
from utils.predict import (
    DISEASES, DISEASE_BY_KEY, list_countries, historical_overall,
    historical_disease, regional_snapshot,
)


def _greeting() -> str:
    h = datetime.now().hour
    if h < 12:  return "Good morning"
    if h < 18:  return "Good afternoon"
    return "Good evening"


def page_header(country_name: str, region: str, date: pd.Timestamp,
               last_updated: str | None = None) -> None:
    """The top strip — greeting + date on the left, location/timestamp right."""
    greet = _greeting()
    date_str = date.strftime("%B %d, %Y") if hasattr(date, "strftime") else str(date)
    last = last_updated or datetime.now().strftime("%b %d, %Y · %I:%M %p")
    st.markdown(
        f"""
<div class="cg-page-head">
  <div class="cg-greet">
    <h1>{greet}, {country_name} 👋</h1>
    <p>Here's your regional environmental health overview for today.</p>
  </div>
  <div class="cg-meta">
    <div class="cg-meta-row">📍 <b>{country_name}</b> · {region}</div>
    <div class="cg-meta-row">📅 {date_str}</div>
    <div class="cg-meta-row">⏱ Last updated: {last}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def disease_card(disease_key: str, level: str | None, score_0_100: float | None,
                 delta_text: str | None = None,
                 spark: list[float] | None = None) -> str:
    """Render a single disease card as HTML.

    ``level``/``score_0_100`` of None renders an explicit "no data" card.
    """
    d = DISEASE_BY_KEY[disease_key]
    if level is None or score_0_100 is None:
        score_str, badge, color, bg = "—", "NO DATA", MUTED, BORDER
    else:
        score_str, badge = f"{score_0_100:.0f}", level.upper()
        color, bg = risk_color(level), RISK_BG[level]
    delta = ""
    if delta_text:
        delta = f'<span style="color:{MUTED};font-size:0.74rem;margin-left:6px">{delta_text}</span>'
    spark_svg = _mini_sparkline(spark) if spark else ""
    return f"""
<div class="cg-disease">
  <div class="cg-disease-head">{d['icon']} {d['label']}</div>
  <div>
    <div class="cg-disease-score" style="color:{INK}">{score_str}</div>
    <div style="margin-top:6px">
      <span class="cg-badge cg-badge-{level}" style="background:{bg};color:{color}">{badge}</span>
      {delta}
    </div>
  </div>
  <div class="cg-disease-foot">
    {spark_svg}
    <span style="font-size:0.72rem;color:{MUTED}">/100</span>
  </div>
</div>
"""


def html_grid(cells: list[str], columns: int, gap: int = 14,
              extra_class: str = "") -> str:
    """Wrap pre-rendered HTML cells in one CSS grid.

    Must be emitted in a single st.markdown call — Streamlit renders each
    call separately, so an opening <div> in one call can't wrap the next.
    Blank lines are stripped so markdown keeps it as one HTML block.
    """
    body = "\n".join(line for c in cells for line in c.splitlines() if line.strip())
    return (f'<div class="{extra_class}" style="display:grid;'
            f'grid-template-columns:repeat({columns},1fr);gap:{gap}px">\n'
            f'{body}\n</div>')


def _mini_sparkline(values: list[float], w: int = 70, h: int = 22) -> str:
    if not values or len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    span = max(1e-6, vmax - vmin)
    pts = []
    for i, v in enumerate(values):
        x = i * (w / (len(values) - 1))
        y = h - ((v - vmin) / span) * h
        pts.append(f"{x:.1f},{y:.1f}")
    last = values[-1]
    color = "#10B981" if last < vmin + 0.4 * span else (
        "#F59E0B" if last < vmin + 0.7 * span else "#EF4444")
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<polyline points="{" ".join(pts)}" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>'
        f'</svg>'
    )


def shap_bars(rows: list[dict]) -> str:
    """Render horizontal SHAP-contribution bars.

    ``rows`` is a list of dicts: {label, magnitude 0..1, contribution_text}.
    """
    if not rows:
        return "<div class='cg-footnote'>No SHAP data available.</div>"
    out = []
    max_mag = max((r["magnitude"] for r in rows), default=1.0) or 1.0
    for r in rows:
        pct = max(0.04, min(1.0, r["magnitude"] / max_mag))
        out.append(f"""
<div class="cg-shap-row">
  <div class="label">{r['label']}</div>
  <div class="bar"><div class="fill" style="width:{pct*100:.1f}%"></div></div>
  <div class="badge">{r.get('contribution_text','')}</div>
</div>
""")
    return "\n".join(out)


def env_cell(icon: str, label: str, value: str, sub: str = "") -> str:
    return f"""
<div class="cg-env-cell">
  <div class="lbl">{icon} {label}</div>
  <div class="val">{value}</div>
  <div class="sub">{sub}</div>
</div>
"""


def regional_profile_card(country: dict, date: pd.Timestamp,
                          n_rows: int) -> str:
    """Replaces the screenshot's personal 'Health Profile' with the
    *regional* profile (we don't have individual medical data)."""
    return f"""
<div class="cg-card">
  <div class="cg-card-head">🏳️ Selected Region Profile</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:8px">
    {env_cell("🌍", "Country",  country["country_name"], "")}
    {env_cell("🗺️", "Region",   country["region"], "")}
    {env_cell("💼", "Income",   country["income_level"], "")}
    {env_cell("🌐", "Climate",  country["climate_zone"], "")}
    {env_cell("👥", "Population", f"{country['population_millions']:.0f}M",
              f"lat {country['latitude']:.1f}, lon {country['longitude']:.1f}")}
    {env_cell("📅", "Data coverage", "2015 – 2025",
              f"{n_rows} test weeks available")}
    {env_cell("🏷️", "ISO code", country["country_code"], "")}
    {env_cell("🔬", "Pipeline", "Phase 6 models", "trained on 2015-2022")}
  </div>
  <div class="cg-footnote" style="margin-top:10px">
    Regional environmental profile, not an individual medical assessment.
    All values come directly from the dataset (no personal health data is
    collected or inferred).
  </div>
</div>
"""
