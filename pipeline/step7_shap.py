"""Step 7 — Phase 7 explainability with SHAP.

For every modelling task we:
1. Re-fit the **best tree-ensemble** model on the training split only.
2. Run ``shap.TreeExplainer`` against the held-out **test** set (~2,350
   future weeks, 2024-2025) — never against train or val.
3. Save two artefacts per task:
   - ``output/shap_importance_<task>.csv`` — global importance: mean |SHAP|
     per feature, with std and the task's own ranking.
   - ``output/shap_summary_<task>.png``     — beeswarm plot (direction) +
     a bar chart of the same means.

Tree-ensemble choice per task (matches Phase 6 selection, but using a tree
ensemble rather than DT/LogReg/ElasticNet — single Decision Trees and
linear models give degenerate SHAP). The brief explicitly says "best
tree-based model (XGBoost/Random Forest)", so for tasks whose Phase-6
winner was a non-tree model (overall_classifier now wins with elastic-net
logistic, not a tree) we fall back to the best tree-ensemble candidate:
- overall_classifier → ``rf``  (best tree ensemble; Phase-6 winner logreg
  is within 0.0002 macro-F1 of rf on validation)
- respiratory        → ``rf``
- vector             → ``rf``
- cardio             → ``xgb``
- waterborne         → ``xgb``
- heat               → ``xgb``

Multiclass SHAP
---------------
``TreeExplainer.shap_values`` for an RF classifier returns a **list** of
arrays, one per class (shap ≤ 0.49 used to; 0.50+ sometimes collapses to
a single 3-D array). We average ``|SHAP|`` across classes to get a single
global-importance number per feature, which is what ``shap.summary_plot
(plot_type='bar')`` does internally.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor

# Reuse step 5's load/split so we operate on exactly the same data.
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
from step5_train_model import (  # noqa: E402
    _load_data,
    _chronological_splits,
    _seed_everything,
    DISEASE_TARGETS,
    COMPOSITE_TARGET,
    TARGET_LABEL,
    TRAIN_END_DATE,
    VAL_END_DATE,
    RANDOM_STATE,
)

_seed_everything(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

OUT_CSV_DIR    = Path("output")
OUT_PNG_DIR    = Path("output/shap")
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)

SHAP_TASKS = [
    ("overall_classifier", "classification", TARGET_LABEL,
     "rf", RandomForestClassifier,
     dict(n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE)),
    ("respiratory", "regression", "respiratory_disease_rate",
     "rf", RandomForestRegressor,
     dict(n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE)),
    ("vector", "regression", "vector_disease_risk_score",
     "rf", RandomForestRegressor,
     dict(n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE)),
    ("cardio", "regression", "cardio_mortality_rate",
     "xgb", XGBRegressor,
     dict(objective="reg:squarederror", n_estimators=300, max_depth=6,
          learning_rate=0.1, tree_method="hist", n_jobs=1, random_state=RANDOM_STATE)),
    ("waterborne", "regression", "waterborne_disease_incidents",
     "xgb", XGBRegressor,
     dict(objective="reg:squarederror", n_estimators=300, max_depth=6,
          learning_rate=0.1, tree_method="hist", n_jobs=1, random_state=RANDOM_STATE)),
    ("heat", "regression", "heat_related_admissions",
     "xgb", XGBRegressor,
     dict(objective="reg:squarederror", n_estimators=300, max_depth=6,
          learning_rate=0.1, tree_method="hist", n_jobs=1, random_state=RANDOM_STATE)),
]

EXPECTED_TOP = {
    "respiratory_disease_rate": ["pm25_ugm3", "pm25_x_aqi", "pm25_ugm3_roll_mean_4w"],
    "vector_disease_risk_score": ["temperature_celsius", "temp_x_pm25",
                                  "precipitation_mm"],
    "heat_related_admissions":   ["temperature_celsius", "heat_wave_days",
                                  "temp_x_pm25"],
    "waterborne_disease_incidents": ["precipitation_mm", "flood_indicator",
                                     "heat_wave_days"],
    "cardio_mortality_rate":     ["pm25_ugm3", "gdp_per_capita_usd",
                                  "healthcare_access_index"],
    "risk_class":                ["temp_x_pm25", "temperature_celsius",
                                  "gdp_per_capita_usd"],
}


def _task_feat_cols(feat_cols, kind, target):
    if kind == "regression":
        drop = set(DISEASE_TARGETS)
    else:
        drop = set()
    drop.add(COMPOSITE_TARGET)
    return [c for c in feat_cols if c != target and c not in drop]


def _aggregate_shap(shap_values, n_features):
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)
        return np.abs(stacked).mean(axis=(0, 1))
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=(0, 2))
    if arr.ndim == 2 and arr.shape[1] == n_features:
        return np.abs(arr).mean(axis=0)
    raise ValueError(f"Unexpected shap_values shape: {arr.shape}")


def _aggregate_shap_std(raw_sv, n_features):
    if isinstance(raw_sv, list):
        stacked = np.stack(raw_sv, axis=0)
        abs_all = np.abs(stacked).reshape(-1, n_features)
        return abs_all.std(axis=0)
    arr = np.asarray(raw_sv)
    if arr.ndim == 3:
        return np.abs(arr).reshape(-1, n_features).std(axis=0)
    if arr.ndim == 2:
        return np.abs(arr).std(axis=0)
    return np.full(n_features, np.nan)


def _shap_for_classification(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    mean_abs = _aggregate_shap(sv, X.shape[1])
    return sv, mean_abs, sv


def _shap_for_regression(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    exp = shap.Explanation(
        values=np.asarray(sv),
        base_values=explainer.expected_value,
        data=X,
        feature_names=list(X.columns),
    )
    mean_abs = _aggregate_shap(sv, X.shape[1])
    return exp, mean_abs, sv


def _plot_pair(task_name, X_test, exp, mean_abs, raw_sv,
               is_classification, out_path):
    order = np.argsort(mean_abs)[::-1]
    sorted_features = X_test.columns[order]
    sorted_means = mean_abs[order]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12),
                             gridspec_kw={"height_ratios": [1, 2]})

    ax_bar = axes[0]
    top = min(15, len(sorted_features))
    ax_bar.barh(range(top)[::-1], sorted_means[:top],
                color="#4575b4", edgecolor="white")
    ax_bar.set_yticks(range(top)[::-1])
    ax_bar.set_yticklabels(sorted_features[:top], fontsize=9)
    ax_bar.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax_bar.set_title(f"{task_name}: global feature importance (top {top})",
                     fontsize=12)
    ax_bar.grid(axis="x", alpha=0.3)

    ax_bee = axes[1]
    plt.sca(ax_bee)
    if is_classification:
        shap.summary_plot(
            raw_sv, X_test, plot_type="dot",
            max_display=15, show=False,
        )
    else:
        shap.plots.beeswarm(exp, max_display=15, show=False, plot_size=None)
    ax_bee.set_title(f"{task_name}: SHAP beeswarm (direction & magnitude)",
                     fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    print("[step7] loading data + chronological splits ...", flush=True)
    X_full, y_frame, feat_cols = _load_data()
    splits = _chronological_splits(X_full)
    train_pos = np.flatnonzero(splits["train"])
    test_pos  = np.flatnonzero(splits["test"])

    train_median = X_full.iloc[train_pos][feat_cols].median(numeric_only=True)

    summary_rows = []
    cross_check = {}

    for task_name, kind, target, model_kind, model_cls, model_kwargs in SHAP_TASKS:
        print(f"\n[step7] === task: {task_name} ({model_kind} on {target}) ===",
              flush=True)

        task_feat_cols = _task_feat_cols(feat_cols, kind, target)
        ok = ~pd.isna(y_frame[target].to_numpy())
        tr_idx = train_pos[ok[train_pos]]
        te_idx = test_pos[ok[test_pos]]

        Xtr = X_full.iloc[tr_idx][task_feat_cols].fillna(train_median)
        Xte = X_full.iloc[te_idx][task_feat_cols].fillna(train_median)
        ytr = y_frame[target].to_numpy()[tr_idx]
        yte = y_frame[target].to_numpy()[te_idx]

        model = model_cls(**model_kwargs)
        t_fit = time.perf_counter()
        model.fit(Xtr, ytr)
        print(f"  fit {model_kind} on n={len(Xtr):,}  ({time.perf_counter()-t_fit:.1f}s)",
              flush=True)

        t_shap = time.perf_counter()
        if model_cls is RandomForestClassifier:
            exp, mean_abs, raw_sv = _shap_for_classification(model, Xte)
        else:
            exp, mean_abs, raw_sv = _shap_for_regression(model, Xte)
        print(f"  shap on n={len(Xte):,} test rows  "
              f"({time.perf_counter()-t_shap:.1f}s)", flush=True)

        imp_df = pd.DataFrame({
            "feature":      Xte.columns,
            "mean_abs_shap": mean_abs,
            "std_abs_shap":  _aggregate_shap_std(raw_sv, Xte.shape[1]),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        imp_df["rank"] = imp_df.index + 1
        csv_path = OUT_CSV_DIR / f"shap_importance_{task_name}.csv"
        imp_df.to_csv(csv_path, index=False)

        png_path = OUT_PNG_DIR / f"shap_summary_{task_name}.png"
        _plot_pair(
            task_name, Xte, exp, mean_abs, raw_sv,
            is_classification=(model_cls is RandomForestClassifier),
            out_path=png_path,
        )

        summary_rows.append({
            "task": task_name,
            "target": target,
            "model": model_kind,
            "n_test": int(len(Xte)),
            "top1_feature": imp_df.iloc[0]["feature"],
            "top1_mean_abs_shap": float(imp_df.iloc[0]["mean_abs_shap"]),
            "csv": str(csv_path),
            "png": str(png_path),
        })

        top10 = imp_df["feature"].head(10).tolist()
        expected = EXPECTED_TOP.get(target, [])
        hits = [f for f in expected if f in top10]
        cross_check[task_name] = {
            "target": target,
            "shap_top10": top10,
            "expected_top": expected,
            "hits_in_top10": hits,
            "hits_count": len(hits),
            "expected_count": len(expected),
        }
        top5_pieces = []
        for i, f in enumerate(top10[:5]):
            top5_pieces.append(f"{f}={imp_df.iloc[i]['mean_abs_shap']:.3f}")
        print("  top-5 SHAP features: " + ", ".join(top5_pieces), flush=True)
        print(f"  EDA cross-check: {len(hits)}/{len(expected)} expected features in SHAP top-10",
              flush=True)

    summary = {
        "shap_version": shap.__version__,
        "test_window": "post 2023-12-31 (held out from training)",
        "n_features_per_task": "see csvs",
        "tasks": summary_rows,
        "eda_cross_check": cross_check,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT_CSV_DIR / "shap_run_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[step7] wrote CSVs   -> {OUT_CSV_DIR}/shap_importance_*.csv", flush=True)
    print(f"[step7] wrote PNGs   -> {OUT_PNG_DIR}/shap_summary_*.png", flush=True)
    print(f"[step7] wrote summary -> {OUT_CSV_DIR}/shap_run_summary.json", flush=True)
    print(f"[step7] runtime      {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
