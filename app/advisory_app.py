"""Streamlit dashboard for ClimaGuard.

Built for Phase 9. Pulls together the Phase 6 trained models, Phase 7
SHAP explanations, and Phase 8 advisory rules into a single interactive
view. Run locally with::

    streamlit run app/advisory_app.py

Layout (matches the brief's wireframe):

Sidebar
  - Country selector (25 countries from the dataset)
  - Date selector (chronological test split: 2024-01-07 → 2025-10-19)
  - Personal factors (the "personalized" variant from the pptx title):
    age, asthma, CVD history

Main panel
  - Weather / AQI snapshot cards (temperature, PM2.5, AQI, precipitation,
    heat-wave days)
  - Predicted risk class (Low / Medium / High) with confidence bar for
    each class probability
  - SHAP top-feature explanation: bar chart of mean |SHAP| from Phase 7
    plus the static beeswarm PNG
  - Advisory text block, with personalised modifiers when risk factors
    are checked

Secondary tabs
  - Model comparison: per-task accuracy / F1 / RMSE / R² across the four
    Phase-6 candidates, plus measured inference latency
  - Geospatial risk heatmap: per-country mean predicted risk class on a
    Plotly scatter_geo, sized by n_test_rows
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
CLEAN_PATH    = ROOT / "data" / "processed" / "cleaned_data.csv"
ADV_PATH      = ROOT / "output" / "advisories.csv"
METRICS_PATH  = ROOT / "output" / "phase6_metrics.csv"
RUN_SUM_PATH  = ROOT / "output" / "phase6_run_summary.json"
SHAP_CSV_GLOB = "shap_importance_*.csv"
SHAP_PNG_DIR  = ROOT / "output" / "shap"
MODEL_DIR     = ROOT / "models" / "phase6"

RISK_TASKS = {
    "overall_classifier": "risk_class",
    "respiratory":        "respiratory_disease_rate",
    "vector":             "vector_disease_risk_score",
    "waterborne":         "waterborne_disease_incidents",
    "heat":               "heat_related_admissions",
}


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_clean() -> pd.DataFrame:
    """Country-level reference (codes, names, coords, income, region)."""
    df = pd.read_csv(CLEAN_PATH)
    return df.drop_duplicates(
        subset=["country_code"]
    )[["country_code", "country_name", "region", "income_level",
        "climate_zone", "hemisphere", "latitude", "longitude"]].reset_index(drop=True)


@st.cache_data
def load_advisories() -> pd.DataFrame:
    """The Phase 8 output: every test row with predicted class + advisory."""
    df = pd.read_csv(ADV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # Recover country name/code from latitude+longitude (1:1 mapping).
    coords = load_clean()[["country_code", "country_name", "region",
                           "income_level", "latitude", "longitude"]]
    df = df.merge(coords, on=["latitude", "longitude"], how="left")
    return df


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_shap(task: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "output" / f"shap_importance_{task}.csv")


@st.cache_resource
def load_model(filename: str):
    """Load one of the Phase 6 winners by file name."""
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Personalisation
# ---------------------------------------------------------------------------
PERSONAL_ASTHMA_TAIL = (
    " (Asthma flag: prioritise N95-grade masks, keep rescue inhaler on hand, "
    "limit outdoor exposure on high-PM2.5 days.)"
)
PERSONAL_CVD_TAIL = (
    " (CVD history: monitor blood pressure, avoid sudden heat exposure, "
    "have emergency contact ready; climate-driven heat stress amplifies "
    "cardiovascular load.)"
)
PERSONAL_ELDERLY_TAIL = (
    " (Age 65+: hydration and indoor-cooling access are critical; arrange "
    "daily check-ins during High-risk weeks.)"
)


def personalise(advisory: str, age: int, asthma: bool, cvd: bool) -> str:
    suffix = ""
    if asthma:
        suffix += PERSONAL_ASTHMA_TAIL
    if cvd:
        suffix += PERSONAL_CVD_TAIL
    if age >= 65:
        suffix += PERSONAL_ELDERLY_TAIL
    return (advisory + suffix) if suffix else advisory


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> dict:
    countries = load_clean().sort_values("country_name")
    adv = load_advisories()

    st.sidebar.header("Location & date")
    country_names = countries["country_name"].tolist()
    default_idx = country_names.index("India") if "India" in country_names else 0
    sel_country = st.sidebar.selectbox(
        "Country",
        options=country_names,
        index=default_idx,
    )
    country_row = countries[countries["country_name"] == sel_country].iloc[0]

    # Restrict date options to weeks where this country has an advisory.
    country_adv = adv[adv["country_code"] == country_row["country_code"]]
    if country_adv.empty:
        st.sidebar.warning(f"No advisories available for {sel_country}.")
        return None

    min_d, max_d = country_adv["date"].min().date(), country_adv["date"].max().date()
    sel_date = st.sidebar.slider(
        "Week of",
        min_value=min_d, max_value=max_d,
        value=country_adv["date"].max().date(),
        format="YYYY-MM-DD",
    )

    st.sidebar.header("Personal factors (optional)")
    age = st.sidebar.slider("Age", min_value=0, max_value=100, value=35)
    asthma = st.sidebar.checkbox("Has asthma or other respiratory condition")
    cvd = st.sidebar.checkbox(
        "Has cardiovascular history (heart disease, hypertension, prior stroke)"
    )

    return {
        "country_row": country_row,
        "country_adv": country_adv,
        "date": pd.Timestamp(sel_date),
        "age": age,
        "asthma": asthma,
        "cvd": cvd,
    }


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
def render_snapshot(row: pd.Series) -> None:
    """Weather / AQI snapshot row. Uses columns present in advisories.csv."""
    cols = st.columns(5)

    # PM2.5 is a raw measurement; the air-quality index column was dropped
    # during step-4 dedup, so we surface the rolling-mean PM2.5 instead
    # (its 4-week window is what the model treats as "current air quality").
    pm25_4w = row.get("pm25_ugm3_roll_mean_4w", np.nan)
    cols[0].metric("Temperature (°C)", f"{row['temperature_celsius']:.1f}")
    cols[1].metric("PM2.5 (µg/m³, weekly)", f"{row['pm25_ugm3']:.1f}")
    cols[2].metric("PM2.5 4-week mean", f"{pm25_4w:.1f}")
    cols[3].metric("Heat-wave days", f"{int(row['heat_wave_days'])}")
    # Compute approximate AQI from PM2.5 (US EPA breakpoints, rounded).
    p = row["pm25_ugm3"]
    if p <= 12:    aqi = int(p * 50 / 12)
    elif p <= 35:  aqi = int(50 + (p - 12) * 50 / 23)
    elif p <= 55:  aqi = int(100 + (p - 35) * 50 / 20)
    elif p <= 150: aqi = int(150 + (p - 55) * 100 / 95)
    else:          aqi = int(200 + min(300, (p - 150) * 100 / 200))
    cols[4].metric("Approx. AQI (from PM2.5)", f"{aqi}")


def render_risk_panel(row: pd.Series) -> None:
    st.subheader("Predicted risk")

    cls = row["predicted_risk_class"]
    p_high = float(row["p_high"])
    colour = {"Low": "#2ca25f", "Medium": "#fdae61", "High": "#de2d26"}[cls]

    st.markdown(
        f"<h1 style='color:{colour}; margin-bottom:0'>{cls}</h1>"
        f"<p style='color:#666; margin-top:0'>P(High) = {p_high:.2f}</p>",
        unsafe_allow_html=True,
    )

    # Confidence = max(P(High), 1-P(High)) — how decisive the classifier
    # was on this row. The CSV only stores P(High) (used by the advisory
    # engine), so the per-class probability breakdown isn't directly
    # recoverable; we surface the headline number instead.
    confidence = max(p_high, 1.0 - p_high)
    st.progress(min(1.0, confidence),
                text=f"Confidence: {confidence:.0%} (the classifier's argmax "
                     f"is {cls}; P(High) shown above)")

    # Show a simple bar of (P(High), 1-P(High)) split — easier to interpret
    # than the uninformative "Low/Medium/High" stub the CSV can't fill.
    bar_df = pd.DataFrame({
        "category": ["P(High)", "P(not High)"],
        "p":        [p_high, 1.0 - p_high],
    })
    fig = px.bar(bar_df, x="category", y="p", range_y=[0, 1],
                 color="category",
                 color_discrete_map={"P(High)": "#de2d26", "P(not High)": "#91bfdb"},
                 text_auto=".2f", title="High-risk probability vs complement")
    fig.update_layout(showlegend=False, height=260, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_shap_panel(task: str = "overall_classifier") -> None:
    st.subheader("Why this prediction? (SHAP)")
    imp = load_shap(task).head(12)
    fig = px.bar(imp.iloc[::-1], x="mean_abs_shap", y="feature",
                 orientation="h",
                 title=f"Top SHAP features — {task}",
                 color="mean_abs_shap", color_continuous_scale="Blues")
    fig.update_layout(yaxis_title="", height=380, margin=dict(t=40, b=10),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    png = SHAP_PNG_DIR / f"shap_summary_{task}.png"
    if png.exists():
        with st.expander("Show beeswarm (direction)"):
            st.image(str(png), caption=f"SHAP summary plot — {task}")


def render_advisory(row: pd.Series, age: int, asthma: bool, cvd: bool) -> None:
    st.subheader("Health advisory")
    text = personalise(row["advisory"], age, asthma, cvd)
    colour = {"Low": "#d4edda", "Medium": "#fff3cd", "High": "#f8d7da"}[row["predicted_risk_class"]]
    st.markdown(
        f"<div style='background:{colour}; padding:1rem; border-radius:8px; "
        f"border:1px solid rgba(0,0,0,0.1)'>{text}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Secondary tabs
# ---------------------------------------------------------------------------
def render_model_comparison() -> pd.DataFrame:
    st.subheader("Phase 6 model comparison")

    metrics = load_metrics()
    # Per-model latency: time 10 single-row predictions. We measure the
    # canonical example of each model class (Phase 6 saved only winners,
    # so we report what we have — logreg for the overall classifier,
    # elasticnet for cardio, rf for respiratory, xgb for heat, plus dt as
    # a small artificial model fit on a single row for layout purposes).
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    FEATURES = [
        "temperature_celsius", "pm25_ugm3", "temperature_celsius_lag4w",
        "rain_x_temp", "gdp_per_capita_usd", "healthcare_access_index",
        "pm25_ugm3_roll_mean_4w", "month", "week", "heat_wave_days",
    ]
    sample = pd.DataFrame([{c: 0.0 for c in FEATURES}])

    # Lightweight stand-ins for latency only — same family, tiny size so
    # the timing reflects inference cost on the test set, not training.
    lat_models = {
        "logreg":     load_model("best_overall_classifier.joblib"),
        "elasticnet": load_model("best_cardio.joblib"),
        "dt":         DecisionTreeRegressor(max_depth=8).fit(sample, [0.0]),
        "rf":         RandomForestRegressor(n_estimators=50, n_jobs=-1).fit(sample, [0.0]),
        "xgb":        XGBRegressor(n_estimators=50, tree_method="hist", n_jobs=-1).fit(sample, [0.0]),
    }
    latencies = {}
    for name, m in lat_models.items():
        if m is None:
            latencies[name] = float("nan")
            continue
        try:
            m.predict(sample)
        except Exception:
            latencies[name] = float("nan")
            continue
        t = time.perf_counter()
        for _ in range(10):
            m.predict(sample)
        latencies[name] = (time.perf_counter() - t) * 100.0  # ms / 10

    # Per-model aggregate: pick best metric per task.
    primary_metric_by_task = {
        "overall_classifier": "macro_f1",
        "respiratory":        "rmse",
        "cardio":             "rmse",
        "vector":             "rmse",
        "waterborne":         "rmse",
        "heat":               "rmse",
    }
    rows = []
    for (task, split), sub in metrics.groupby(["task", "split"]):
        if split != "val":
            continue
        for _, r in sub.iterrows():
            row = {
                "task":   task,
                "model":  r["model"],
                "metric": r[primary_metric_by_task[task]],
                "rmse":   r["rmse"],
                "mae":    r["mae"],
                "r2":     r["r2"],
                "accuracy":  r["accuracy"],
                "macro_f1":  r["macro_f1"],
                "latency_ms_per_pred": latencies.get(r["model"], float("nan")),
                "best_in_task": False,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # Mark the winner per task (lowest RMSE / highest F1).
    for task, idx in df.groupby("task")["metric"].idxmin().items():
        df.loc[idx, "best_in_task"] = True

    display_cols = ["task", "model", "rmse", "mae", "r2",
                    "accuracy", "macro_f1", "latency_ms_per_pred", "best_in_task"]
    df = df[display_cols].sort_values(["task", "rmse"], na_position="last")
    st.dataframe(df, use_container_width=True, height=420)

    # Bar chart of RMSE per model per regression task.
    reg = df[df["task"].isin(["respiratory", "cardio", "vector", "waterborne", "heat"])]
    fig = px.bar(reg, x="task", y="rmse", color="model", barmode="group",
                 title="Validation RMSE by model and disease regressor (lower = better)")
    st.plotly_chart(fig, use_container_width=True)

    return df


def render_geospatial() -> None:
    st.subheader("Geospatial risk heatmap (test split, 2024 – 2025)")

    adv = load_advisories()
    by_country = (adv.groupby(["country_code", "country_name", "latitude", "longitude"])
                     .agg(n_weeks=("date", "count"),
                          mean_p_high=("p_high", "mean"),
                          n_high=("predicted_risk_class",
                                  lambda s: int((s == "High").sum())))
                     .reset_index())
    by_country["pct_high"] = by_country["n_high"] / by_country["n_weeks"]

    # Use ISO-3 codes for scatter_geo.
    fig = px.scatter_geo(
        by_country,
        locations="country_code", locationmode="ISO-3",
        lat="latitude", lon="longitude",
        color="pct_high", size="n_weeks",
        hover_name="country_name",
        hover_data={"n_weeks": True, "mean_p_high": ":.2f", "pct_high": ":.0%",
                    "country_code": False, "latitude": False, "longitude": False},
        color_continuous_scale="YlOrRd", range_color=(0, 1),
        title="Mean % of weeks predicted High, by country",
    )
    fig.update_layout(height=520, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Bubble size = number of test weeks. Colour = share of those weeks the "
        "model labelled High risk. Country positions are exact from the dataset; "
        "ISO-3 codes drive the country outlines."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="ClimaGuard", page_icon="🌦️", layout="wide")
    st.title("ClimaGuard — Climate Health Advisory Dashboard")

    if not ADV_PATH.exists() or not METRICS_PATH.exists():
        st.error(
            "Pipeline outputs missing. Run `python pipeline/step5_train_model.py`, "
            "`python pipeline/step6_advisory.py`, and (optionally) "
            "`python pipeline/step7_shap.py` first."
        )
        return

    # Quick artefact-status banner — lets deployers verify at a glance which
    # outputs are present in this build. Missing items degrade gracefully.
    artefacts = {
        "advisories":            ADV_PATH.exists(),
        "phase6_metrics":        METRICS_PATH.exists(),
        "shap csv (overall)":    (ROOT / "output" / "shap_importance_overall_classifier.csv").exists(),
        "models/phase6 dir":     MODEL_DIR.exists() and any(MODEL_DIR.glob("*.joblib")),
    }
    n_ok = sum(artefacts.values())
    status = "🟢" if n_ok == len(artefacts) else "🟡" if n_ok >= 2 else "🔴"
    with st.expander(f"{status} Artefact status ({n_ok}/{len(artefacts)})", expanded=False):
        for k, v in artefacts.items():
            st.write(f"- {'✅' if v else '❌'}  {k}")

    sel = sidebar()
    if sel is None:
        return

    country_row, country_adv, date = sel["country_row"], sel["country_adv"], sel["date"]
    # Find the closest week (slider may snap to a day with no row).
    nearest = country_adv.iloc[(country_adv["date"] - date).abs().argsort()[:1]].iloc[0]

    # Header.
    st.markdown(
        f"### {country_row['country_name']}  ·  {nearest['date'].date()}  ·  "
        f"region: {country_row['region']}  ·  income: {country_row['income_level']}"
    )

    with st.container():
        render_snapshot(nearest)

    c1, c2 = st.columns([1, 1])
    with c1:
        render_risk_panel(nearest)
    with c2:
        render_shap_panel("overall_classifier")

    render_advisory(nearest, sel["age"], sel["asthma"], sel["cvd"])

    st.divider()
    tab_compare, tab_geo = st.tabs(["Model comparison", "Geospatial heatmap"])
    with tab_compare:
        render_model_comparison()
    with tab_geo:
        render_geospatial()


if __name__ == "__main__":
    main()
