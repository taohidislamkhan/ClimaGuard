"""Visual theme + CSS for ClimaGuard.

Dark-navy sidebar, light main canvas, cyan accents, rounded cards. Designed
to match the reference screenshot's colour palette and spacing.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Colour palette (single source of truth — used by every chart + card).
# ---------------------------------------------------------------------------
NAVY        = "#0F2447"
NAVY_DEEP   = "#0A1A36"
NAVY_SOFT   = "#1B3358"
CYAN        = "#22D3EE"
CYAN_DARK   = "#0EA5E9"
TEAL        = "#14B8A6"
INK         = "#0F172A"
INK_SOFT    = "#334155"
MUTED       = "#64748B"
BG_PAGE     = "#F1F5F9"   # the light blue-gray page background
BG_CARD     = "#FFFFFF"
BORDER      = "#E2E8F0"

# Risk semantics — used everywhere. Never swap.
RISK_GREEN  = "#10B981"
RISK_AMBER  = "#F59E0B"
RISK_RED    = "#EF4444"

RISK_BG = {
    "Low":    "#ECFDF5",
    "Medium": "#FEF3C7",
    "High":   "#FEE2E2",
}
RISK_FG = {
    "Low":    RISK_GREEN,
    "Medium": RISK_AMBER,
    "High":   RISK_RED,
}


def risk_color(level: str) -> str:
    return RISK_FG.get(level, MUTED)


# ---------------------------------------------------------------------------
# CSS injected once at the top of main().
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Page background ---------- */
html, body, [data-testid="stAppViewContainer"], .main {{
    background: {BG_PAGE};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: {INK};
}}

/* Hide Streamlit's default page chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{display: none;}}

/* ---------- Sidebar — dark navy ---------- */
section[data-testid="stSidebar"] {{
    background: {NAVY};
    color: #E2E8F0;
    width: 260px !important;
    min-width: 260px !important;
    border-right: 1px solid {NAVY_SOFT};
}}
section[data-testid="stSidebar"] * {{
    color: #E2E8F0;
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {{
    color: #CBD5E1;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.08);
}}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: {NAVY_DEEP};
    border-color: rgba(255,255,255,0.1);
    color: #F1F5F9;
}}
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {{
    background: {CYAN};
    border-color: {CYAN};
}}

/* ---------- Cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] > div,
.stCard {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(15, 36, 71, 0.06),
                0 1px 2px rgba(15, 36, 71, 0.04);
    padding: 0.6rem 0.8rem;
}}

/* The block container for our cards uses an even subtler default. */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px;
}}

/* Tighter, slightly larger section headings */
h1 {{ font-size: 1.65rem; font-weight: 700; letter-spacing: -0.01em; color: {INK}; }}
h2 {{ font-size: 1.20rem; font-weight: 700; letter-spacing: -0.01em; color: {INK}; }}
h3 {{ font-size: 1.00rem; font-weight: 600; letter-spacing: -0.005em; color: {INK}; }}

/* Sidebar brand block */
.cg-brand {{
    padding: 18px 16px 8px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.cg-brand .cg-brand-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.cg-brand .cg-brand-sub {{
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 2px;
}}

/* Sidebar nav rows */
.cg-nav {{padding: 8px 8px 16px 8px;}}
.cg-nav-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 10px;
    font-size: 0.92rem;
    color: #CBD5E1;
    cursor: pointer;
    transition: background 120ms ease, color 120ms ease;
}}
.cg-nav-row:hover {{
    background: rgba(255,255,255,0.04);
    color: #F1F5F9;
}}
.cg-nav-row.active {{
    background: rgba(34, 211, 238, 0.10);
    color: {CYAN};
    font-weight: 600;
}}
.cg-nav-row .cg-nav-icon {{ font-size: 1.05rem; width: 22px; text-align: center; }}
.cg-nav-sep {{
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 8px 12px;
}}

/* Sidebar footer status block */
.cg-status {{
    position: relative;
    padding: 14px 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 0.82rem;
    color: #94A3B8;
}}
.cg-status .dot {{
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    background: {RISK_GREEN};
    margin-right: 8px;
    vertical-align: middle;
    box-shadow: 0 0 6px {RISK_GREEN};
}}

/* ---------- Page header strip ---------- */
.cg-page-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 4px 18px 4px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 22px;
}}
.cg-page-head .cg-greet h1 {{margin: 0 0 4px 0;}}
.cg-page-head .cg-greet p {{margin: 0; color: {MUTED}; font-size: 0.95rem;}}
.cg-page-head .cg-meta {{
    text-align: right;
    font-size: 0.85rem;
    color: {MUTED};
}}
.cg-page-head .cg-meta .cg-meta-row {{
    display: flex; align-items: center; gap: 6px; justify-content: flex-end;
    margin-bottom: 4px;
}}

/* ---------- Hero risk card ---------- */
.cg-hero {{
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 28px;
    align-items: center;
    padding: 20px 24px;
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    box-shadow: 0 4px 16px rgba(15, 36, 71, 0.06);
}}
.cg-hero .cg-hero-score {{
    position: relative;
    display: flex; align-items: center; justify-content: center;
    height: 180px;
}}
.cg-hero .cg-hero-num {{
    position: absolute;
    text-align: center;
    font-weight: 700;
    color: {INK};
}}
.cg-hero .cg-hero-num .big {{
    font-size: 3.4rem; line-height: 1;
    letter-spacing: -0.02em;
}}
.cg-hero .cg-hero-num .sub {{
    font-size: 0.9rem; color: {MUTED}; margin-top: 4px;
}}
.cg-hero .cg-hero-body h2 {{ margin: 0 0 6px 0; }}
.cg-hero .cg-hero-body .cg-hero-tag {{ margin: 6px 0 12px 0; }}
.cg-hero .cg-hero-body .cg-hero-meta {{
    color: {MUTED}; font-size: 0.92rem; line-height: 1.45; margin-bottom: 14px;
}}

/* Risk badge (pill) */
.cg-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}}
.cg-badge-Low    {{ background: {RISK_BG['Low']};    color: {RISK_FG['Low']};    }}
.cg-badge-Medium {{ background: {RISK_BG['Medium']}; color: {RISK_FG['Medium']}; }}
.cg-badge-High   {{ background: {RISK_BG['High']};   color: {RISK_FG['High']};   }}

/* Big number metrics */
.cg-metric-label {{ color: {MUTED}; font-size: 0.82rem; font-weight: 500; }}
.cg-metric-value {{ font-size: 1.6rem; font-weight: 700; color: {INK}; letter-spacing: -0.01em; }}
.cg-metric-sub   {{ color: {MUTED}; font-size: 0.78rem; margin-top: 2px; }}

/* Disease card */
.cg-disease {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 14px 16px;
    height: 130px;
    display: flex; flex-direction: column; justify-content: space-between;
}}
.cg-disease .cg-disease-head {{
    display: flex; align-items: center; gap: 8px;
    color: {INK_SOFT}; font-size: 0.88rem; font-weight: 600;
}}
.cg-disease .cg-disease-score {{
    font-size: 1.85rem; font-weight: 700; letter-spacing: -0.01em;
    line-height: 1; color: {INK};
}}
.cg-disease .cg-disease-foot {{
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.78rem; color: {MUTED};
}}

/* SHAP contribution bars */
.cg-shap-row {{
    display: grid;
    grid-template-columns: 130px 1fr 70px;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
}}
.cg-shap-row .label {{ color: {INK}; font-weight: 500; font-size: 0.9rem; }}
.cg-shap-row .bar  {{
    height: 10px; border-radius: 6px; background: #F1F5F9; position: relative;
    overflow: hidden;
}}
.cg-shap-row .bar .fill {{
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #FCD34D 0%, #FB923C 50%, #EF4444 100%);
}}
.cg-shap-row .badge {{
    text-align: right; font-size: 0.78rem; font-weight: 600;
}}

/* Section card with subtle accent */
.cg-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px 20px;
}}
.cg-card-head {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 12px;
    color: {INK};
    font-size: 1.0rem; font-weight: 600;
}}
.cg-card-sub {{ color: {MUTED}; font-size: 0.85rem; margin-top: -8px; margin-bottom: 12px; }}

/* Inline disclaimer — small, italic */
.cg-footnote {{
    color: {MUTED};
    font-size: 0.78rem;
    font-style: italic;
    line-height: 1.4;
}}

/* Callout (advisory) */
.cg-callout {{
    border-radius: 12px;
    padding: 12px 14px;
    border: 1px solid;
    font-size: 0.92rem;
    line-height: 1.5;
}}
.cg-callout-low    {{ background: {RISK_BG['Low']};    border-color: #A7F3D0; color: #065F46; }}
.cg-callout-medium {{ background: {RISK_BG['Medium']}; border-color: #FDE68A; color: #92400E; }}
.cg-callout-high   {{ background: {RISK_BG['High']};   border-color: #FECACA; color: #991B1B; }}

/* Hero illustration block on the right (decorative) */
.cg-illu {{
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    height: 180px;
    background: linear-gradient(180deg, #BAE6FD 0%, #E0F2FE 60%, #DCFCE7 100%);
    border: 1px solid {BORDER};
}}
.cg-illu .sun {{
    position: absolute; top: 16px; right: 28px;
    width: 40px; height: 40px; border-radius: 50%;
    background: radial-gradient(circle, #FDE68A 0%, #F59E0B 100%);
    box-shadow: 0 0 24px rgba(245, 158, 11, 0.5);
}}
.cg-illu svg {{ position: absolute; bottom: 0; left: 0; right: 0; width: 100%; }}

/* Pill (segmented tabs) */
.cg-pills {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.cg-pill {{
    padding: 5px 12px; border-radius: 999px;
    background: #F1F5F9; color: {INK_SOFT};
    font-size: 0.82rem; font-weight: 500;
    border: 1px solid transparent;
    cursor: pointer;
}}
.cg-pill.active {{ background: {CYAN}; color: white; border-color: {CYAN}; }}

/* Recalculate button */
.cg-recalc-btn button {{
    background: {CYAN_DARK};
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
}}
.cg-recalc-btn button:hover {{ background: #0284C7; }}

/* View-Detailed-Analysis primary button */
.cg-cta button {{
    background: {NAVY};
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 8px 14px;
    font-weight: 600;
}}

/* Environment mini metric grid */
.cg-env-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.cg-env-cell {{
    background: #F8FAFC;
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px 12px;
    text-align: left;
}}
.cg-env-cell .lbl {{ color: {MUTED}; font-size: 0.74rem; font-weight: 500; }}
.cg-env-cell .val {{ color: {INK}; font-size: 1.0rem; font-weight: 700; margin-top: 2px; }}
.cg-env-cell .sub {{ color: {MUTED}; font-size: 0.72rem; margin-top: 2px; }}

/* Hide the dataframes' default index column visually when we want clean look */
.cg-clean table tbody th {{ display: none; }}
</style>
"""


def inject() -> None:
    """Call once from main() at the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small SVG used in the hero illustration block.
# ---------------------------------------------------------------------------
HERO_ILLU_SVG = """
<svg viewBox="0 0 320 90" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <!-- far skyline -->
  <g fill="#94A3B8" opacity="0.55">
    <rect x="10"  y="40" width="22" height="50"/>
    <rect x="36"  y="30" width="14" height="60"/>
    <rect x="54"  y="50" width="20" height="40"/>
    <rect x="78"  y="20" width="10" height="70"/>
    <rect x="92"  y="44" width="18" height="46"/>
    <rect x="114" y="34" width="14" height="56"/>
    <rect x="132" y="52" width="22" height="38"/>
    <rect x="158" y="22" width="12" height="68"/>
    <rect x="174" y="48" width="20" height="42"/>
    <rect x="198" y="30" width="14" height="60"/>
    <rect x="216" y="56" width="22" height="34"/>
    <rect x="242" y="38" width="14" height="52"/>
    <rect x="260" y="48" width="20" height="42"/>
    <rect x="284" y="30" width="14" height="60"/>
    <rect x="302" y="50" width="14" height="40"/>
  </g>
  <!-- mid trees -->
  <g fill="#16A34A">
    <circle cx="32"  cy="76" r="10"/>
    <circle cx="40"  cy="72" r="8"/>
    <circle cx="124" cy="78" r="12"/>
    <circle cx="132" cy="74" r="9"/>
    <circle cx="220" cy="80" r="11"/>
    <circle cx="228" cy="76" r="8"/>
  </g>
  <!-- ground -->
  <rect x="0" y="86" width="320" height="4" fill="#16A34A"/>
</svg>
"""


# ---------------------------------------------------------------------------
# Component: ring score (SVG donut)
# ---------------------------------------------------------------------------
def render_ring_score(score: float, level: str, size: int = 200) -> str:
    """Circular risk-score donut, color follows the level."""
    color = risk_color(level)
    radius = (size - 16) / 2
    circ = 2 * 3.14159 * radius
    pct = max(0.0, min(1.0, score / 100.0))
    dash = circ * pct
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none"
          stroke="#E2E8F0" stroke-width="14"/>
  <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none"
          stroke="{color}" stroke-width="14"
          stroke-dasharray="{dash:.1f} {circ:.1f}"
          stroke-linecap="round"
          transform="rotate(-90 {size/2} {size/2})"/>
</svg>
"""
