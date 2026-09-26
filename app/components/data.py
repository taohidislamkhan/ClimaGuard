"""Cached data + model loaders and shared constants.

Everything that touches disk lives here so the page modules can just call
``load_meta()`` etc. without re-reading CSVs on every rerun.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR  = Path(__file__).resolve().parent.parent
ROOT     = APP_DIR.parent
CLEAN_PATH    = ROOT / "data" / "processed" / "cleaned_data.csv"
ADV_PATH      = ROOT / "output" / "advisories.csv"
METRICS_PATH  = ROOT / "output" / "phase6_metrics.csv"
FEAT_SEL      = ROOT / "output" / "feature_selection.csv"
FEAT_ENG_META = ROOT / "output" / "feature_engineering_meta.json"
FEAT_SEL_META = ROOT / "output" / "feature_selection_meta.json"
SHAP_PNG_DIR  = ROOT / "output" / "shap"
CHARTS_DIR    = ROOT / "output" / "charts"
MODEL_DIR     = ROOT / "models" / "phase6"

# Tab 2/3 risk targets (keyed by the same names used everywhere).
RISK_TARGETS = [
    ("overall_classifier", "Overall risk",       "risk_class"),
    ("respiratory",        "Respiratory",        "respiratory_disease_rate"),
    ("cardio",             "Cardiovascular",     "cardio_mortality_rate"),
    ("vector",             "Vector-borne",       "vector_disease_risk_score"),
    ("waterborne",         "Waterborne",         "waterborne_disease_incidents"),
    ("heat",               "Heat-related",       "heat_related_admissions"),
]

# Color semantics — used consistently across every chart.
RISK_COLOR = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}

AQI_BREAKPOINTS = [  # (pm25_lo, pm25_hi, aqi_lo, aqi_hi, label, color)
    (0.0,  12.0,   0,  50,  "Good",                  "#2E7D32"),
    (12.0, 35.4,  51, 100,  "Moderate",              "#F9A825"),
    (35.4, 55.4, 101, 150,  "Unhealthy for Sensitive","#EF6C00"),
    (55.4, 150.4,151, 200,  "Unhealthy",             "#C62828"),
    (150.4,250.4,201, 300,  "Very Unhealthy",        "#6A1B9A"),
    (250.4,500.4,301, 500,  "Hazardous",             "#880E4F"),
]

DISCLAIMER = (
    "_Regional nowcast, not a medical diagnosis — this dashboard "
    "forecasts population-level risk for a given country and week, "
    "and is not advice about any individual's health._"
)

# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_clean() -> pd.DataFrame:
    """Country-level reference (codes, names, coords, income, region)."""
    df = pd.read_csv(CLEAN_PATH)
    return (
        df.drop_duplicates(subset=["country_code"])
        .reset_index(drop=True)[
            ["country_code", "country_name", "region", "income_level",
             "climate_zone", "hemisphere", "latitude", "longitude"]
        ]
    )


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_shap(task: str) -> pd.DataFrame:
    """Top-features table for one task."""
    return pd.read_csv(ROOT / "output" / f"shap_importance_{task}.csv")


@st.cache_data
def load_feat_selection() -> pd.DataFrame:
    return pd.read_csv(FEAT_SEL)


@st.cache_data
def load_meta() -> dict:
    out = {}
    if FEAT_ENG_META.exists():
        out["engineering"] = json.loads(FEAT_ENG_META.read_text())
    if FEAT_SEL_META.exists():
        out["selection"] = json.loads(FEAT_SEL_META.read_text())
    return out


# ---------------------------------------------------------------------------
# Lookup helpers (cheap, no caching needed)
# ---------------------------------------------------------------------------
def aq_from_pm25(pm: float) -> tuple[int, str, str]:
    """Return (aqi, label, color) from PM2.5 µg/m³ (US EPA breakpoints)."""
    pm = max(0.0, float(pm))
    for lo, hi, aqi_lo, aqi_hi, label, color in AQI_BREAKPOINTS:
        if pm <= hi:
            if hi == lo:
                aqi = aqi_lo
            else:
                aqi = int(round(aqi_lo + (pm - lo) * (aqi_hi - aqi_lo) / (hi - lo)))
            return aqi, label, color
    return 500, "Hazardous", "#880E4F"


def risk_color(level: str) -> str:
    return RISK_COLOR.get(level, "#9E9E9E")


def country_options() -> list[str]:
    return sorted(load_clean()["country_name"].dropna().unique().tolist())


def has_artifacts() -> dict[str, bool]:
    """Artefact presence — used by the top-of-page banner."""
    return {
        "advisories":         ADV_PATH.exists(),
        "phase6_metrics":     METRICS_PATH.exists(),
        "shap csv (overall)": (ROOT / "output" / "shap_importance_overall_classifier.csv").exists(),
        "models/phase6 dir":  MODEL_DIR.exists() and any(MODEL_DIR.glob("*.joblib")),
    }
