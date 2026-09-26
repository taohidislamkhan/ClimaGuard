"""Dark-navy sidebar with full nav and footer status."""

from __future__ import annotations

import streamlit as st

from components.theme import inject
from utils.predict import list_countries, get_date_range


NAV_GROUPS = [
    # (label, icon, page_key) — the active page is set in main() via st.session_state.
    ("Dashboard",          "🏠", "dashboard"),
    ("My Risk",            "📊", "my_risk"),
    ("Environment",        "🌦️", "environment"),
    ("Risk Map",           "🗺️", "risk_map"),
    ("Risk History",       "📈", "risk_history"),
]
NAV_GROUPS_2 = [
    ("Regional Analysis",  "🌍", "regional"),
    ("Disease Analysis",   "🦠", "disease"),
    ("Model Performance",  "🤖", "models"),
    ("Explainability",     "🔍", "explain"),
]
NAV_GROUPS_3 = [
    ("Data & Methodology", "📊", "methodology"),
    ("How It Works",       "❓", "how"),
    ("Settings",           "⚙️", "settings"),
]

ALL_PAGES = [p for group in (NAV_GROUPS, NAV_GROUPS_2, NAV_GROUPS_3)
             for _, _, p in group]


def _nav_row(label: str, icon: str, key: str, current: str) -> str:
    active = "active" if key == current else ""
    return (
        f'<div class="cg-nav-row {active}" '
        f'onclick="document.title=\'cg-nav:{key}\'">'
        f'<span class="cg-nav-icon">{icon}</span>'
        f'<span>{label}</span></div>'
    )


def render(active_page: str) -> dict:
    """Render the full sidebar. Returns the user's current selection."""
    # Use st.session_state to track the current page (so clicking nav
    # actually navigates — Streamlit reruns the script on widget change).
    st.session_state.setdefault("page", active_page)

    with st.sidebar:
        st.markdown(
            '<div class="cg-brand">'
            '<div class="cg-brand-title">🌦️ ClimaGuard</div>'
            '<div class="cg-brand-sub">Regional Environmental Disease Risk Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="cg-nav">', unsafe_allow_html=True)
        for label, icon, key in NAV_GROUPS:
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True):
                st.session_state["page"] = key
                st.rerun()
        st.markdown('<div class="cg-nav-sep"></div>', unsafe_allow_html=True)
        for label, icon, key in NAV_GROUPS_2:
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True):
                st.session_state["page"] = key
                st.rerun()
        st.markdown('<div class="cg-nav-sep"></div>', unsafe_allow_html=True)
        for label, icon, key in NAV_GROUPS_3:
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True):
                st.session_state["page"] = key
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Footer status block
        st.markdown(
            '<div class="cg-status">'
            '<div><span class="dot"></span><b>System Online</b></div>'
            '<div style="margin-top:8px">Data updated</div>'
            '<div>2015-01-04 → 2025-10-19</div>'
            '<div style="margin-top:6px; color:#64748B">Last run: Phase 1–7 pipeline</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Inputs live below the nav in the sidebar (country + date)
    sel = _inputs()
    return {"page": st.session_state["page"], **sel}


def _inputs() -> dict:
    """Country + date inputs. Stored in session_state so all pages see them."""
    countries = list_countries()
    by_code = {c["country_code"]: c for c in countries}
    codes = [c["country_code"] for c in countries]

    if "country" not in st.session_state:
        st.session_state["country"] = "BGD" if "BGD" in codes else codes[0]
    country = st.sidebar.selectbox(
        "Country",
        options=codes,
        index=codes.index(st.session_state["country"]),
        format_func=lambda c: by_code[c]["country_name"],
        key="country_select",
    )
    st.session_state["country"] = country

    mn, mx = get_date_range(country)
    if "date" not in st.session_state:
        st.session_state["date"] = mx
    d = st.sidebar.date_input(
        "Week of",
        value=st.session_state["date"].date() if hasattr(st.session_state["date"], "date")
              else st.session_state["date"],
        min_value=mn.date(), max_value=mx.date(),
        key="date_input",
    )
    import pandas as pd
    st.session_state["date"] = pd.Timestamp(d)

    return {
        "country_code": country,
        "country": by_code[country],
        "date": st.session_state["date"],
    }
