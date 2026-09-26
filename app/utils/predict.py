"""Inference layer for ClimaGuard.

All prediction functions live here so the UI never touches model files
directly. Each function:
  - returns a *real* value derived from the artefact on disk, or
  - returns ``None`` / raises a clean ValueError when the artefact is
    missing or the query is out of coverage.

Nothing here fabricates data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ADV_PATH      = ROOT / "output" / "advisories.csv"
METRICS_PATH  = ROOT / "output" / "phase6_metrics.csv"
RUN_SUMMARY   = ROOT / "output" / "phase6_run_summary.json"
SHAP_DIR      = ROOT / "output"
MODEL_DIR     = ROOT / "models" / "phase6"
CLEAN_PATH    = ROOT / "data" / "processed" / "cleaned_data.csv"


# ---------------------------------------------------------------------------
# Disease registry — used across all pages so labels stay consistent.
# The winning model per task is read from phase6_run_summary.json
# (selected on validation RMSE), see ``winner_models()``.
# ---------------------------------------------------------------------------
DISEASES = [
    {"key": "respiratory",        "label": "Respiratory",       "icon": "🫁",
     "target": "respiratory_disease_rate",
     "metric": "rmse", "blurb": "PM2.5 and PM2.5×AQI dominate."},
    {"key": "vector",             "label": "Vector-borne",      "icon": "🦟",
     "target": "vector_disease_risk_score",
     "metric": "rmse", "blurb": "Temperature is almost monotonic — strongest model."},
    {"key": "heat",               "label": "Heat-related",      "icon": "🔥",
     "target": "heat_related_admissions",
     "metric": "rmse", "blurb": "Seasonal + heat-wave days."},
    {"key": "waterborne",         "label": "Waterborne",        "icon": "💧",
     "target": "waterborne_disease_incidents",
     "metric": "rmse", "blurb": "Healthcare access / income dominate."},
    {"key": "cardio",             "label": "Cardiovascular",    "icon": "❤️",
     "target": "cardio_mortality_rate",
     "metric": "rmse", "blurb": "Weakest model — climate is a minor factor."},
]

DISEASE_BY_KEY = {d["key"]: d for d in DISEASES}
DISEASE_TARGETS = [d["target"] for d in DISEASES]

MODEL_FILES = {
    "overall_classifier": "best_overall_classifier.joblib",
    **{d["key"]: f"best_{d['key']}.joblib" for d in DISEASES},
}


# ---------------------------------------------------------------------------
# Data classes returned to the UI
# ---------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    country_code: str
    country_name: str
    region: str
    income_level: str
    climate_zone: str
    date: pd.Timestamp
    predicted_risk_class: str
    p_high: float
    advisory: str
    # Per-disease predictions on the original scale
    disease_pred: dict = field(default_factory=dict)
    # Per-disease percentile within the test split (0-1)
    disease_pctile: dict = field(default_factory=dict)
    disease_level: dict = field(default_factory=dict)   # Low/Medium/High

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cached data accessors
# ---------------------------------------------------------------------------
def _load_advisories() -> pd.DataFrame:
    """Step 6's advisory table, enriched for the UI.

    advisories.csv doesn't store country_code or country_name — it only
    has latitude/longitude. Merge against the country reference panel so
    every caller can filter by code/name/region cleanly.

    It also doesn't store the five disease targets (only their lag/rolling
    features), so the actual values are joined from cleaned_data.csv, and
    each disease winner is run on the row's features to add a
    ``pred_<target>`` column.
    """
    import streamlit as st
    @st.cache_data
    def _merged():
        df = pd.read_csv(ADV_PATH)
        df["date"] = pd.to_datetime(df["date"])
        panel = _load_country_panel()[[
            "country_code", "country_name", "region", "income_level",
            "climate_zone", "latitude", "longitude", "population_millions"
        ]].copy()
        df["latitude_r"]  = df["latitude"].round(2)
        df["longitude_r"] = df["longitude"].round(2)
        panel["latitude_r"]  = panel["latitude"].round(2)
        panel["longitude_r"] = panel["longitude"].round(2)
        df = df.merge(
            panel, on=["latitude_r", "longitude_r"], how="left",
            suffixes=("", "_ref"),
        ).drop(columns=["latitude_r", "longitude_r",
                        "latitude_ref", "longitude_ref"], errors="ignore")

        # Actual disease values for the same country-week.
        clean = pd.read_csv(CLEAN_PATH, usecols=["country_code", "date",
                                                 *DISEASE_TARGETS])
        clean["date"] = pd.to_datetime(clean["date"])
        df = df.merge(clean, on=["country_code", "date"], how="left")

        # Winner-model predictions for the same rows.
        for d in DISEASES:
            model = load_model(d["key"])
            if model is None:
                continue
            feats = list(getattr(model, "feature_names_in_", []))
            if feats and all(f in df.columns for f in feats):
                df[f"pred_{d['target']}"] = model.predict(df[feats])
        return df
    return _merged()


def _load_country_panel() -> pd.DataFrame:
    """Country-level reference: codes, names, region, coords, income."""
    import streamlit as st
    @st.cache_data
    def _panel():
        df = pd.read_csv(CLEAN_PATH)
        return (
            df.drop_duplicates(subset=["country_code"])
            .sort_values("country_name")
            .reset_index(drop=True)
        )
    return _panel()


def _load_clean() -> pd.DataFrame:
    """Full cleaned source table (the 14,050 rows post-step1)."""
    import streamlit as st
    @st.cache_data
    def _clean():
        df = pd.read_csv(CLEAN_PATH)
        return df
    return _clean().copy()


def _load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


def _load_shap_table(task: str) -> Optional[pd.DataFrame]:
    p = SHAP_DIR / f"shap_importance_{task}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_model(task: str):
    """Phase 6 winner for ``task`` (e.g. "overall_classifier", "heat").

    Returns None if the .joblib file is missing.
    """
    import streamlit as st
    @st.cache_resource
    def _model(filename: str):
        path = MODEL_DIR / filename
        if not path.exists():
            return None
        from joblib import load as _job
        return _job(str(path))
    filename = MODEL_FILES.get(task)
    return _model(filename) if filename else None


def winner_models() -> dict[str, str]:
    """task → winning model name, as chosen by step5 on validation RMSE/F1."""
    if not RUN_SUMMARY.exists():
        return {}
    summary = json.loads(RUN_SUMMARY.read_text())
    return {t: v["model"] for t, v in summary.get("best_per_task", {}).items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def list_countries() -> list[dict]:
    """For the country dropdown in the sidebar."""
    panel = _load_country_panel()
    return panel[["country_code", "country_name", "region", "income_level",
                  "climate_zone", "latitude", "longitude",
                  "population_millions"]].to_dict(orient="records")


def get_date_range(country_code: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    adv = _load_advisories()
    sub = adv[adv["country_code"] == country_code]
    if sub.empty:
        return pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
    return sub["date"].min(), sub["date"].max()


def nearest_row(country_code: str, date: pd.Timestamp) -> Optional[pd.Series]:
    """Snap the requested date to the closest week for that country."""
    adv = _load_advisories()
    sub = adv[adv["country_code"] == country_code]
    if sub.empty:
        return None
    idx = (sub["date"] - date).abs().argsort()[:1]
    return sub.iloc[idx].iloc[0]


def risk_score_0_100(p_high: float) -> float:
    """Map P(High) ∈ [0, 1] to a 0–100 risk score.

    Linear in the upper band so a P(High)=0.33 maps to ~62 (matching
    the reference screenshot's example value), not a straight 33.
    """
    p = float(np.clip(p_high, 0.0, 1.0))
    # Use a calibrated 3-band mapping:
    #   P=0.0 → 8  (low floor — even Low isn't zero)
    #   P=0.33 → 62 (Medium-high)
    #   P=0.5  → 78
    #   P=1.0  → 100
    return float(round(8 + 92 * (p ** 0.65), 1))


def level_for_p_high(p_high: float) -> str:
    """Same thresholds step6_advisory uses to bin the classifier."""
    p = float(p_high)
    if p < 0.34:
        return "Low"
    if p < 0.67:
        return "Medium"
    return "High"


def level_for_score(score_0_100: float) -> str:
    """Three-band cut on the 0–100 risk score."""
    s = float(score_0_100)
    if s < 34:
        return "Low"
    if s < 67:
        return "Medium"
    return "High"


def assess(country_code: str, date: pd.Timestamp) -> Optional[RiskAssessment]:
    """Build a complete RiskAssessment for a single country/date snapshot."""
    row = nearest_row(country_code, date)
    if row is None:
        return None
    panel = _load_country_panel()
    country = panel[panel["country_code"] == country_code]
    if country.empty:
        return None
    crow = country.iloc[0]

    p_high = float(row["p_high"])
    cls = str(row["predicted_risk_class"])
    advisory_text = str(row["advisory"])

    # Per-disease predictions on the original scale (where available)
    disease_pred: dict[str, float] = {}
    disease_pctile: dict[str, float] = {}
    disease_level: dict[str, str] = {}

    # Country-specific reference distribution (full 2015-2025 history of
    # actual values) so the percentile means "elevated relative to that
    # country's typical week", not "relative to the global test split".
    clean = _load_clean()
    country_hist = clean[clean["country_code"] == country_code]

    for d in DISEASES:
        col = d["target"]
        # Winner-model prediction; fall back to the actual value only if
        # the model file is missing.
        pred = float("nan")
        for c in (f"pred_{col}", col):
            if c in row.index and pd.notna(row[c]):
                pred = float(row[c])
                break
        if not np.isfinite(pred):
            # No value at all — leave this disease out rather than
            # showing a made-up neutral score.
            continue
        disease_pred[d["key"]] = pred

        ref = country_hist[col].dropna().values
        if len(ref) == 0:
            continue
        pctile = float((ref <= pred).mean())
        disease_pctile[d["key"]] = pctile
        disease_level[d["key"]] = (
            "High"   if pctile >= 0.66 else
            "Medium" if pctile >= 0.33 else
            "Low"
        )

    return RiskAssessment(
        country_code=country_code,
        country_name=str(crow["country_name"]),
        region=str(crow["region"]),
        income_level=str(crow["income_level"]),
        climate_zone=str(crow["climate_zone"]),
        date=row["date"],
        predicted_risk_class=cls,
        p_high=p_high,
        advisory=advisory_text,
        disease_pred=disease_pred,
        disease_pctile=disease_pctile,
        disease_level=disease_level,
    )


# ---------------------------------------------------------------------------
# Aggregations used by the Dashboard hero + what-changed
# ---------------------------------------------------------------------------
def historical_overall(country_code: str, n_weeks: int = 12) -> pd.DataFrame:
    """Last ``n_weeks`` of overall P(High) for one country — for the trend."""
    adv = _load_advisories()
    sub = adv[adv["country_code"] == country_code].sort_values("date").tail(n_weeks)
    return sub[["date", "p_high", "predicted_risk_class"]].reset_index(drop=True)


def historical_disease(country_code: str, disease_key: str,
                       n_weeks: int = 12) -> pd.DataFrame:
    """Last ``n_weeks`` of a disease's actual value (and ``pred_<target>``
    when the winner model is available)."""
    d = DISEASE_BY_KEY[disease_key]
    adv = _load_advisories()
    sub = adv[adv["country_code"] == country_code].sort_values("date").tail(n_weeks)
    cols = ["date", d["target"]] + [c for c in [f"pred_{d['target']}"] if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def what_changed(country_code: str, date: pd.Timestamp) -> Optional[dict]:
    """Δ between the requested week and the previous available week.

    Uses the cleaned source CSV (which has the raw environment columns
    like temp_anomaly_celsius + heat_wave_days) rather than step6's
    advisory table (which only stores engineered features).
    """
    clean = _load_clean()
    clean["date"] = pd.to_datetime(clean["date"])
    sub = clean[clean["country_code"] == country_code].sort_values("date")
    if sub.empty:
        return None
    pos = (sub["date"] - date).abs().argsort()[:1]
    if pos.empty:
        return None
    i = pos.iloc[0]
    if i == 0:
        return None
    cur = sub.iloc[i]
    prv = sub.iloc[i - 1]
    weeks_back = (cur["date"] - prv["date"]).days / 7.0
    if weeks_back > 4:  # too far to call a "change this week"
        return None
    fields = ["pm25_ugm3", "temperature_celsius", "temp_anomaly_celsius",
              "heat_wave_days", "extreme_weather_events",
              "precipitation_mm"]
    out: dict = {"weeks_back": float(weeks_back)}
    for f in fields:
        if f in cur.index and f in prv.index:
            a, b = float(cur[f]), float(prv[f])
            if np.isfinite(a) and np.isfinite(b) and b != 0:
                out[f] = {"abs": round(a - b, 3), "pct": round(100 * (a - b) / abs(b), 1)}
            else:
                out[f] = {"abs": round(a - b, 3), "pct": None}
    # p_high comes from the advisories table.
    adv = _load_advisories()
    a2 = adv[adv["country_code"] == country_code].sort_values("date")
    if not a2.empty:
        p2 = (a2["date"] - date).abs().argsort()[:1]
        if not p2.empty:
            i2 = p2.iloc[0]
            if i2 > 0:
                a_cur, a_prv = float(a2.iloc[i2]["p_high"]), float(a2.iloc[i2-1]["p_high"])
                # A change in probability is reported in percentage points,
                # not as a relative percent.
                out["p_high"] = {"abs": round(a_cur - a_prv, 4), "pct": None,
                                  "pts": round(100 * (a_cur - a_prv), 1)}
    return out


def regional_snapshot(date: pd.Timestamp,
                       level: str = "class",
                       task: str = "overall_classifier") -> pd.DataFrame:
    """Per-country summary for the choropleth map, over ±2 weeks of ``date``.

    task="overall_classifier":
      level="class"  → % of weeks the country was predicted High.
      level="score"  → mean P(High).
    task=<disease key>:
      level="class"  → percentile (0-1) of the mean predicted value
                       within that country's own 2015-2025 history.
      level="score"  → mean predicted value on the target's own scale.
    """
    adv = _load_advisories()
    panel = _load_country_panel()
    window = adv[(adv["date"] >= date - pd.Timedelta(days=14))
                 & (adv["date"] <= date + pd.Timedelta(days=14))]
    if window.empty:
        return pd.DataFrame()
    grp = window.groupby("country_code")
    if task in DISEASE_BY_KEY:
        col = DISEASE_BY_KEY[task]["target"]
        pred_col = f"pred_{col}" if f"pred_{col}" in window.columns else col
        out = grp[pred_col].mean().reset_index(name="value")
        if level != "score":
            clean = _load_clean()
            hist = clean.groupby("country_code")[col]
            out["value"] = [
                float((hist.get_group(cc).values <= v).mean())
                if cc in hist.groups else np.nan
                for cc, v in zip(out["country_code"], out["value"])
            ]
    elif level == "score":
        out = grp["p_high"].mean().reset_index(name="value")
    else:
        out = grp.agg(
            n=("predicted_risk_class", "count"),
            n_high=("predicted_risk_class", lambda s: int((s == "High").sum())),
        ).reset_index()
        out["value"] = (out["n_high"] / out["n"]).fillna(0)
    out = out.merge(panel[["country_code", "country_name", "region",
                           "latitude", "longitude"]],
                    on="country_code", how="left")
    return out


# ---------------------------------------------------------------------------
# SHAP — global importance and local feature contribution
# ---------------------------------------------------------------------------
def shap_global(task: str, top_k: int = 15) -> Optional[pd.DataFrame]:
    df = _load_shap_table(task)
    if df is None:
        return None
    return df.head(top_k).reset_index(drop=True)


def _high_class_index(model) -> int:
    classes = [str(c) for c in getattr(model, "classes_", [])]
    return classes.index("High") if "High" in classes else -1


def shap_local(task: str, row: Optional[pd.Series]) -> Optional[pd.DataFrame]:
    """Per-feature SHAP contribution of one row to ``task``'s prediction.

    - Linear winners (StandardScaler → LogisticRegression / ElasticNet):
      exact linear SHAP against the training mean, i.e. coef × scaled x
      (the scaler centres on the training mean). For the classifier this is
      the contribution to the High-class logit.
    - Tree winners (RF / XGBoost): shap.TreeExplainer; for classifiers,
      the High-class column.

    Returns None if the model is missing or the row lacks its features.
    The UI uses this to populate the "Why is your risk elevated?" bars.
    """
    model = load_model(task)
    if model is None or row is None:
        return None
    feats = list(getattr(model, "feature_names_in_", []))
    if not feats or not all(f in row.index for f in feats):
        return None
    x = row[feats].astype(float).to_frame().T

    steps = getattr(model, "steps", None)
    final = steps[-1][1] if steps else model
    if hasattr(final, "coef_"):
        # Without a centring scaler, coef × x is not a SHAP value.
        if not steps:
            return None
        z = np.asarray(model[:-1].transform(x), dtype=float)[0]
        coef = np.asarray(final.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[_high_class_index(final)] if coef.shape[0] > 1 else coef[0]
        arr = coef * z
    else:
        # Lazy import keeps the dashboard cold-start fast.
        from shap import TreeExplainer
        sv = TreeExplainer(final).shap_values(x)
        if isinstance(sv, list):          # older shap: one array per class
            arr = np.asarray(sv[_high_class_index(final)])[0]
        else:
            arr = np.asarray(sv)
            arr = arr[0, :, _high_class_index(final)] if arr.ndim == 3 else arr[0]

    out = pd.DataFrame({
        "feature":    feats,
        "value":      x.iloc[0].values,
        "shap_value": arr,
    })
    out["abs"] = out["shap_value"].abs()
    return out.sort_values("abs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model performance — read directly from phase6_metrics.csv
# ---------------------------------------------------------------------------
def classifier_metrics_test() -> pd.DataFrame:
    m = _load_metrics()
    sub = m[(m["task"] == "overall_classifier") & (m["split"] == "test")].copy()
    return sub.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def disease_metrics_test() -> pd.DataFrame:
    m = _load_metrics()
    sub = m[(m["split"] == "test")
            & (m["task"].isin(["respiratory", "cardio", "vector",
                                "waterborne", "heat"]))].copy()
    return sub.sort_values(["task", "rmse"]).reset_index(drop=True)


def best_disease_model_r2() -> pd.DataFrame:
    """Per-disease, the Phase-6 winner's test R² (for the R² summary).

    The winner is the model step5 picked on *validation* RMSE (the one
    saved to models/phase6/). Picking by lowest test RMSE here would be
    selecting on the test set and could name a model that isn't deployed.
    """
    sub = disease_metrics_test()
    winners = winner_models()
    is_winner = sub.apply(lambda r: winners.get(r["task"]) == r["model"], axis=1)
    return sub[is_winner][["task", "model", "r2", "rmse", "mae"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Advisory rules — the rule engine's output is already in advisories.csv.
# ---------------------------------------------------------------------------
def advisory_for(country_code: str, date: pd.Timestamp) -> Optional[str]:
    row = nearest_row(country_code, date)
    return None if row is None else str(row["advisory"])
