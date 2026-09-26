"""Settings page — runtime diagnostics, data freshness, artefact status."""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from components.layout import page_header
from components.data import has_artifacts, load_meta
from utils.predict import (
    classifier_metrics_test, disease_metrics_test, get_date_range,
    list_countries,
)


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    st.markdown("### ⚙️ Settings & Diagnostics")

    # Artefact status
    st.markdown("#### Artefact status")
    artefacts = has_artifacts()
    for k, v in artefacts.items():
        st.write(f"- {'✅' if v else '❌'}  **{k}**")

    # Pipeline metadata
    st.markdown("#### Pipeline metadata")
    meta = load_meta()
    st.json({
        "engineering": {k: v for k, v in meta.get("engineering", {}).items()
                        if k != "composite_cols"},
        "selection":   {k: v for k, v in meta.get("selection", {}).items()
                        if k not in ("committee", "examples_dropped_pairs",
                                     "top10_by_mean_rank")},
    })

    # Runtime
    st.markdown("#### Runtime")
    st.write(f"- Python: {platform.python_version()}")
    st.write(f"- Platform: {platform.system()} {platform.machine()}")

    # Coverage
    st.markdown("#### Data coverage")
    for c in list_countries():
        mn, mx = get_date_range(c["country_code"])
        st.write(f"- **{c['country_name']}** ({c['country_code']}) — "
                 f"{mn.date()} → {mx.date()}")
