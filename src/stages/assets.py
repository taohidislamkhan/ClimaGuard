"""Stage 9 - assets: precompute the dashboard's static page data into ``artifacts/*.json``.

    dvc repro assets        (or: python -m src.stages.assets [--only shap metrics])

Everything is derived from pipeline outputs — the raw and cleaned data,
``data/processed/features.parquet``, ``reports/`` and the saved winners in
``models/``. Nothing here is hand-typed.

Outputs
-------
dataset.json           size, coverage, nulls, duplicates
cleaning.json          counts behind each cleaning rule (raw vs cleaned)
correlation.json       Pearson matrix of the key variables (cleaned data)
eda.json               temperature-vs-heat scatter + bins, monthly seasonal indices
features.json          feature-family counts and the selection funnel
metrics.json           all 24 models x val/test from phase6_metrics.csv + winners
shap_global.json       mean |SHAP| of each saved winner (from reports/shap, grouped labels)
test_predictions.json  predicted vs actual per disease on the test split
country_map.json       model scores per country per test week (Risk Map)
bgd_history.json       model scores for every Bangladesh week (Risk History)
seasonality.json       month x disease average percentile of observed values
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from dashboard import scoring, shap_utils
from dashboard.inference import MODEL_NAMES, TARGETS, ModelStore
from src.core.features import drop_first_week
from src.utils import paths
from src.utils.config import load_params

warnings.filterwarnings("ignore")

PARAMS = load_params()
ART = paths.ARTIFACTS
TRAIN_END_DATE = PARAMS["split"]["train_end"]
VAL_END = PARAMS["split"]["val_end"]
SEED = PARAMS["seed"]
SAMPLES = PARAMS["assets"]
DISEASES = ["respiratory", "vector", "heat", "waterborne", "cardio"]
KEY_VARS = ["temperature_celsius", "precipitation_mm", "heat_wave_days", "pm25_ugm3",
            "air_quality_index", "healthcare_access_index", "gdp_per_capita_usd",
            "respiratory_disease_rate", "cardio_mortality_rate", "vector_disease_risk_score",
            "waterborne_disease_incidents", "heat_related_admissions"]
VAR_LABELS = {
    "temperature_celsius": "Temperature", "precipitation_mm": "Precipitation",
    "heat_wave_days": "Heat-wave days", "pm25_ugm3": "PM2.5", "air_quality_index": "AQI",
    "healthcare_access_index": "Healthcare access", "gdp_per_capita_usd": "GDP per capita",
    "respiratory_disease_rate": "Respiratory rate", "cardio_mortality_rate": "Cardio mortality",
    "vector_disease_risk_score": "Vector risk", "waterborne_disease_incidents": "Waterborne cases",
    "heat_related_admissions": "Heat admissions",
}


def write(name: str, obj) -> None:
    ART.mkdir(exist_ok=True)
    path = ART / name
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    print(f"  wrote {path.relative_to(paths.ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ---------------------------------------------------------------------------
def build_dataset_and_cleaning(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    write("dataset.json", {
        "rows": len(raw), "columns": raw.shape[1],
        "countries": int(raw["country_name"].nunique()),
        "regions": int(raw["region"].nunique()),
        "region_names": sorted(raw["region"].unique().tolist()),
        "start": raw["date"].min(), "end": raw["date"].max(),
        "weeks_per_country": int(raw.groupby("country_name").size().median()),
        "nulls": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_country_weeks": int(raw.duplicated(["country_code", "date"]).sum()),
        "clean_rows": len(clean),
    })
    targets = list(TARGETS.values())
    write("cleaning.json", {
        "negative_aqi": int((raw["air_quality_index"] < 0).sum()),
        "negative_aqi_after": int((clean["air_quality_index"] < 0).sum()),
        "healthcare_over_100": int((raw["healthcare_access_index"] > 100).sum()),
        "healthcare_over_100_after": int((clean["healthcare_access_index"] > 100).sum()),
        "negative_disease_values": {c: int((raw[c] < 0).sum()) for c in targets},
        "negative_disease_values_after": {c: int((clean[c] < 0).sum()) for c in targets},
        # step 1 keeps the first row per (country, year, week); ISO week
        # numbering repeats a week at some year boundaries.
        "duplicate_year_week_rows": int(raw.duplicated(["country_code", "year", "week"]).sum()),
        "rows_removed": len(raw) - len(clean),
    })


def build_correlation_and_eda(clean: pd.DataFrame) -> None:
    corr = clean[KEY_VARS].corr().round(3)
    write("correlation.json", {"vars": KEY_VARS, "labels": [VAR_LABELS[v] for v in KEY_VARS],
                               "matrix": corr.values.tolist()})

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(clean), size=min(SAMPLES["eda_sample"], len(clean)), replace=False)
    sample = clean.iloc[idx]
    bins = np.arange(np.floor(clean["temperature_celsius"].min()), clean["temperature_celsius"].max() + 2, 2)
    cut = pd.cut(clean["temperature_celsius"], bins)
    binned = clean.groupby(cut, observed=True)["heat_related_admissions"].agg(["mean", "count"])
    binned = binned[binned["count"] >= 20]

    clean = clean.assign(month=pd.to_datetime(clean["date"]).dt.month)
    seasonal = {}
    for scope, df in [("all", clean), ("bgd", clean[clean["country_code"] == "BGD"])]:
        m = df.groupby("month")[list(TARGETS.values())].mean()
        idxd = (100 * m / m.mean()).round(1)          # index: annual mean = 100
        seasonal[scope] = {k: idxd[t].tolist() for k, t in TARGETS.items()}
    write("eda.json", {
        "scatter": {"temp": sample["temperature_celsius"].round(2).tolist(),
                    "heat": sample["heat_related_admissions"].round(2).tolist()},
        "bins": {"mid": [r(i.mid, 1) for i in binned.index], "mean": binned["mean"].round(2).tolist(),
                 "count": binned["count"].astype(int).tolist()},
        "seasonal_index": seasonal,
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    })


def build_features(store: ModelStore, cols: list[str]) -> None:
    meta = json.loads(paths.FEATURE_SELECTION.read_text())
    sel = pd.read_csv(paths.RANKINGS)
    fam = {
        "Rolling windows (4/8/12 w)": sum("_roll_" in c for c in cols),
        "Lags (1/2/4/8 w)": sum("_lag" in c for c in cols),
        "Cyclical sin/cos": sum(c.endswith(("_sin", "_cos")) for c in cols),
        "Interactions": sum("_x_" in c for c in cols),
        "One-hot dummies": sum(c.startswith(("country_code_", "region_", "income_level_", "climate_zone_")) for c in cols),
    }
    write("features.json", {
        "families": fam,
        "funnel": [
            {"stage": "Engineered columns", "n": len(cols)},
            {"stage": "Numeric candidates", "n": meta["candidate_features"]},
            {"stage": f"After |r| > {meta['corr_threshold']} de-duplication", "n": meta["after_dedup"]},
            {"stage": f"Committee vote (≥{meta['votes_needed']} of 3 rankings, top {PARAMS['select']['top_k']} cap)",
             "n": meta["survivors_with_min_votes"]},
            {"stage": "Used by the models (targets removed)", "n": len(store.features)},
        ],
        "methods": [
            {"name": "Correlation filter", "detail": f"drop one of each pair with |r| > {meta['corr_threshold']}"},
            {"name": "Pearson correlation", "detail": "rank by |r| with the target"},
            {"name": "Mutual information", "detail": "rank by non-linear dependence"},
            {"name": "Random Forest importance", "detail": "rank by impurity importance"},
        ],
        "votes_top": sel.sort_values("mean_rank").head(10)[["feature", "votes", "mean_rank"]].to_dict("records"),
        "model_features": store.features,
    })


def build_metrics() -> dict:
    m = pd.read_csv(paths.MODEL_COMPARISON)
    winners = json.loads(paths.VALIDATION_METRICS.read_text())["winners"]
    rows = json.loads(m.to_json(orient="records"))
    write("metrics.json", {"rows": rows, "winners": winners})
    return winners


def build_shap_global() -> None:
    """Group the explain stage's per-feature mean |SHAP| under friendly labels."""
    out = {}
    for task in ["overall"] + list(TARGETS):
        rep = json.loads((paths.SHAP / f"{'overall_classifier' if task == 'overall' else task}.json").read_text())
        df = pd.DataFrame(rep["importance"])
        df["label"] = df["feature"].map(shap_utils.friendly_label)
        grouped = df.groupby("label")["mean_abs"].sum().sort_values(ascending=False)
        out[task] = {
            "model": MODEL_NAMES[task], "n_rows": rep["n_rows"],
            "features": df.head(12).assign(mean_abs=lambda d: d["mean_abs"].round(4))[["feature", "label", "mean_abs"]]
                          .to_dict("records"),
            "grouped": [{"label": k, "mean_abs": round(float(v), 4)} for k, v in grouped.head(8).items()],
            "unit": rep["unit"],
        }
    write("shap_global.json", out)


def build_predictions(store: ModelStore, fd: pd.DataFrame, X: pd.DataFrame, names: dict) -> None:
    preds = store.predict(X)
    proba = pd.DataFrame([p.proba for p in preds])
    dis = pd.DataFrame([p.diseases for p in preds])
    scores = pd.DataFrame({k: [store.percentile(k, v) for v in dis[k]] for k in DISEASES})
    scores["overall"] = [scoring.overall_score(p) for p in proba.to_dict("records")]
    level = [scoring.overall_level(p) for p in proba.to_dict("records")]
    dates = fd["date"].dt.date.astype(str)

    # Test split: predicted vs actual
    test = (fd["date"] > VAL_END).to_numpy()
    rng = np.random.default_rng(SEED)
    tp = {}
    for k in DISEASES:
        y, yhat = fd.loc[test, TARGETS[k]].to_numpy(float), dis.loc[test, k].to_numpy(float)
        ss_res, ss_tot = ((y - yhat) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        pick = rng.choice(len(y), size=min(SAMPLES["scatter_sample"], len(y)), replace=False)
        bgd = test & (fd["country_name"] == "Bangladesh").to_numpy()
        tp[k] = {
            "target": TARGETS[k], "model": MODEL_NAMES[k], "n": int(len(y)),
            "r2": r(1 - ss_res / ss_tot), "rmse": r(np.sqrt(((y - yhat) ** 2).mean())),
            "scatter": {"actual": np.round(y[pick], 2).tolist(), "pred": np.round(yhat[pick], 2).tolist()},
            "bgd": {"dates": dates[bgd].tolist(),
                    "actual": fd.loc[bgd, TARGETS[k]].round(2).tolist(),
                    "pred": dis.loc[bgd, k].round(2).tolist()},
        }
    write("test_predictions.json", {"start": dates[test].min(), "end": dates[test].max(), "targets": tp})

    # Country view of the Risk Map: every country, every test week
    weeks = sorted(dates[test].unique().tolist())
    cmap = {}
    for code, name in names.items():
        mask = test & (fd["country_name"] == name).to_numpy()
        sub = scores[mask].assign(date=dates[mask].values, level=np.array(level)[mask]).set_index("date")
        sub = sub.reindex(weeks)
        cmap[code] = {"name": name,
                      "overall": [None if pd.isna(v) else round(v) for v in sub["overall"]],
                      "level": [None if pd.isna(v) else v for v in sub["level"]],
                      **{k: [None if pd.isna(v) else round(v) for v in sub[k]] for k in DISEASES}}
    write("country_map.json", {"weeks": weeks, "countries": cmap})

    # Every Bangladesh week, for Risk History
    b = (fd["country_name"] == "Bangladesh").to_numpy()
    write("bgd_history.json", {
        "dates": dates[b].tolist(), "level": np.array(level)[b].tolist(),
        **{k: scores.loc[b, k].round(1).tolist() for k in ["overall"] + DISEASES},
        "split": ["train" if d <= TRAIN_END_DATE else "val" if d <= VAL_END else "test" for d in dates[b]],
    })


def build_seasonality(store: ModelStore, fd: pd.DataFrame) -> None:
    fd = fd.assign(month=fd["date"].dt.month)
    out = {}
    for scope, df in [("bgd", fd[fd["country_name"] == "Bangladesh"]), ("all", fd)]:
        rows = {}
        for k in DISEASES:
            pct = [store.percentile(k, v) for v in df[TARGETS[k]]]
            rows[k] = pd.Series(pct, index=df.index).groupby(df["month"]).mean().round(1).tolist()
        out[scope] = rows
    write("seasonality.json", {"months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                          "Sep", "Oct", "Nov", "Dec"],
                               "values": out,
                               "metric": "average percentile (0–100) of the observed weekly value "
                                         "within the 2015–2022 training distribution"})


STEPS = ["dataset", "eda", "features", "metrics", "shap", "predictions", "seasonality"]


def main(only: list[str] | None = None) -> None:
    steps = set(only or STEPS)
    t0 = time.time()
    print("[assets] loading data and models ...")
    raw = pd.read_csv(paths.RAW_CSV)
    clean = pd.read_parquet(paths.CLEAN)
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    store = ModelStore()
    fd = pd.read_parquet(paths.FEATURES)
    cols = list(fd.columns)
    fd = drop_first_week(fd)
    X = store.to_matrix(fd)
    names = dict(raw[["country_code", "country_name"]].drop_duplicates().values)

    if "dataset" in steps:
        build_dataset_and_cleaning(raw, clean)
    if "eda" in steps:
        build_correlation_and_eda(clean)
    if "features" in steps:
        build_features(store, cols)
    if "metrics" in steps:
        build_metrics()
    if "shap" in steps:
        build_shap_global()
    if "predictions" in steps:
        build_predictions(store, fd, X, names)
    if "seasonality" in steps:
        build_seasonality(store, fd)
    print(f"[assets] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=STEPS, help="rebuild only these assets")
    main(ap.parse_args().only)
