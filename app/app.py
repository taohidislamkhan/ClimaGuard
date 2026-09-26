"""ClimaGuard — top-level orchestrator.

Launch::

    streamlit run app/app.py

This file just wires the sidebar + page router. Pages live in app/pages/.
"""

from __future__ import annotations

import streamlit as st

from components.theme import inject
from components.sidebar import render as render_sidebar
from utils.predict import _load_advisories  # noqa: F401  (warm up cache)


PAGE_MAP = {
    "dashboard":   ("Dashboard",         "pages.dashboard"),
    "my_risk":     ("My Risk",           "pages.my_risk"),
    "environment": ("Environment",       "pages.environment"),
    "risk_map":    ("Risk Map",          "pages.risk_map"),
    "risk_history":("Risk History",      "pages.risk_history"),
    "regional":    ("Regional Analysis", "pages.regional"),
    "disease":     ("Disease Analysis",  "pages.disease"),
    "models":      ("Model Performance", "pages.models"),
    "explain":     ("Explainability",    "pages.explain"),
    "methodology": ("Data & Methodology","pages.methodology"),
    "how":         ("How It Works",      "pages.how"),
    "settings":    ("Settings",          "pages.settings"),
}


def main() -> None:
    st.set_page_config(
        page_title="ClimaGuard — Regional Environmental Disease Risk Intelligence",
        page_icon="🌦️", layout="wide",
    )
    inject()

    # Sidebar (navigation + inputs)
    state = render_sidebar(active_page=st.session_state.get("page", "dashboard"))

    # Page router
    page_key = state["page"]
    page_label, page_module = PAGE_MAP[page_key]
    mod = __import__(page_module, fromlist=["render"])
    mod.render(state)


if __name__ == "__main__":
    main()
