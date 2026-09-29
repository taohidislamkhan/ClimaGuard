"""Local SHAP explanations with friendly, grouped feature labels.

The overall classifier (scaled Logistic Regression) gets exact linear SHAP
against the training mean — coef × standardised x — for the High-class
logit. Tree models (RF / XGBoost) use ``shap.TreeExplainer``.

Engineered columns are grouped under one friendly label (all PM2.5 lags and
rolling windows -> "PM2.5") and their SHAP values summed, which keeps SHAP's
additivity. Only model features can appear; humidity is not one.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

FRIENDLY = [
    (r"^temp_x_pm25$",                   "Temperature × PM2.5"),
    (r"^pm25_x_aqi$",                    "PM2.5 × Air Quality Index"),
    (r"^rain_x_temp$",                   "Rainfall × Temperature"),
    (r"^pm25_ugm3",                      "PM2.5"),
    (r"^air_quality_index",              "Air Quality Index"),
    (r"^temperature_celsius",            "Temperature"),
    (r"^precipitation_mm",               "Rainfall"),
    (r"^heat_wave_days",                 "Heat-wave days"),
    (r"^waterborne_disease_incidents",   "Recent waterborne cases"),
    (r"^heat_related_admissions",        "Recent heat admissions"),
    (r"^(week|week_sin|week_cos|month)$", "Season (week of year)"),
    (r"^gdp_per_capita_usd$",            "GDP per capita"),
    (r"^healthcare_access_index$",       "Healthcare access"),
    (r"^food_security_index$",           "Food security"),
    (r"^(latitude|longitude)$",          "Geographic location"),
    (r"^(region_|income_level_|climate_zone_)", "Country context"),
]


# Groups built from lags / rolling windows of a disease target itself.
AUTOREGRESSIVE = {"Recent waterborne cases", "Recent heat admissions"}


def friendly_label(feature: str) -> str:
    for pattern, label in FRIENDLY:
        if re.search(pattern, feature):
            return label
    return feature


def _high_index(model) -> int:
    classes = [str(c) for c in getattr(model, "classes_", [])]
    return classes.index("High") if "High" in classes else -1


_EXPLAINERS: dict[int, object] = {}


def tree_explainer(model):
    """TreeExplainer per model, built once (construction dominates the cost)."""
    key = id(model)
    if key not in _EXPLAINERS:
        from shap import TreeExplainer   # lazy: only tree models need it
        _EXPLAINERS[key] = TreeExplainer(model)
    return _EXPLAINERS[key]


def local_shap(model, x: pd.DataFrame) -> np.ndarray:
    """Per-feature SHAP values for a single-row frame ``x``."""
    steps = getattr(model, "steps", None)
    final = steps[-1][1] if steps else getattr(model, "model", model)   # unwrap EncodedClassifier
    if hasattr(final, "coef_") and steps:
        z = np.asarray(model[:-1].transform(x), dtype=float)[0]
        coef = np.asarray(final.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[_high_index(model)] if coef.shape[0] > 1 else coef[0]
        return coef * z
    sv = tree_explainer(final).shap_values(x)
    if isinstance(sv, list):
        return np.asarray(sv[_high_index(model)])[0]
    arr = np.asarray(sv)
    return arr[0, :, _high_index(model)] if arr.ndim == 3 else arr[0]


def contribution_level(norm: float) -> str:
    if norm >= 66:
        return "High"
    if norm >= 33:
        return "Moderate"
    return "Low"


def grouped_signed(model, x: pd.DataFrame, k: int = 8) -> list[dict]:
    """Top-k grouped factors with signed SHAP values (waterfall-style charts)."""
    sv = local_shap(model, x)
    df = pd.DataFrame({"feature": x.columns, "shap": sv})
    df["label"] = df["feature"].map(friendly_label)
    g = (df.groupby("label").agg(shap=("shap", "sum"), features=("feature", list))
           .reset_index())
    g = g[g["shap"].abs() > 1e-9]                  # linear models zero out many features
    g = g.reindex(g["shap"].abs().sort_values(ascending=False).index).head(k)
    return [{"label": r["label"], "shap": round(float(r["shap"]), 4), "features": r["features"],
             "autoregressive": r["label"] in AUTOREGRESSIVE} for _, r in g.iterrows()]


def top_factors(model, x: pd.DataFrame, k: int = 6) -> list[dict]:
    """Top-k grouped factors, bar length = |SHAP| normalised to 0–100."""
    sv = local_shap(model, x)
    df = pd.DataFrame({"feature": x.columns, "shap": sv})
    df["label"] = df["feature"].map(friendly_label)
    grouped = (df.groupby("label")
                 .agg(shap=("shap", "sum"), features=("feature", list))
                 .reset_index())
    grouped["abs"] = grouped["shap"].abs()
    grouped = grouped.sort_values("abs", ascending=False).head(k)
    top = float(grouped["abs"].max()) or 1.0
    out = []
    for _, r in grouped.iterrows():
        norm = round(100.0 * r["abs"] / top)
        out.append({
            "label": r["label"],
            "value": norm,
            "shap": float(r["shap"]),
            "direction": "raises" if r["shap"] > 0 else "lowers",
            "level": contribution_level(norm),
            "features": r["features"],
            "autoregressive": r["label"] in AUTOREGRESSIVE,
        })
    return out
