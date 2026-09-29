"""Every file the pipeline writes and the dashboard reads, in one place."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "params.yaml"

# data
RAW_CSV = ROOT / "data" / "raw" / "global_climate_health_impact_tracker_2015_2025.csv"
CLEAN = ROOT / "data" / "interim" / "clean.parquet"
PROCESSED = ROOT / "data" / "processed"
FEATURES = PROCESSED / "features.parquet"
SELECTED = PROCESSED / "selected_features.json"
RANKINGS = PROCESSED / "feature_rankings.csv"
SPLITS = {s: PROCESSED / f"{s}.parquet" for s in ("train", "val", "test")}

# models
MODELS = ROOT / "models"
CANDIDATES = MODELS / "candidates"
PREPROCESS = MODELS / "preprocess.json"      # feature list, train medians, class labels

# reports
REPORTS = ROOT / "reports"
METRICS = REPORTS / "metrics"
PLOTS = REPORTS / "plots"
SHAP = REPORTS / "shap"
DRIFT = REPORTS / "drift"
DATA_QUALITY = METRICS / "data_quality.json"
FEATURIZE_SUMMARY = METRICS / "featurize.json"
FEATURE_SELECTION = METRICS / "feature_selection.json"
SPLIT_SUMMARY = METRICS / "split_summary.json"
VALIDATION_METRICS = METRICS / "validation_metrics.json"
CLASSIFIER_METRICS = METRICS / "classifier_metrics.json"
REGRESSOR_METRICS = METRICS / "regressor_metrics.json"
MODEL_COMPARISON = METRICS / "model_comparison.csv"
DRIFT_REPORT = DRIFT / "drift_report.json"

# dashboard assets
ARTIFACTS = ROOT / "artifacts"


def best_model(task: str) -> Path:
    return MODELS / f"best_{task}.joblib"


def candidate(task: str, model: str) -> Path:
    return CANDIDATES / f"{task}__{model}.joblib"
