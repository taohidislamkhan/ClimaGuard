"""Model loading and feature-row construction.

The dashboard never trains. It loads the validation winners that the DVC
``train`` stage saved as ``models/best_<task>.joblib`` (which model won each
task is read from ``reports/metrics/validation_metrics.json``).

Every model expects the same columns (``feature_names_in_``). A live
feature row is built in three layers:

1. A **template**: the latest Bangladesh week in ``features.parquet``.
   It supplies the country context (GDP, healthcare access, food security,
   lat/lon, one-hot region/income/climate) and the lagged disease counts
   (waterborne / heat-admission lags), which no live API provides.
2. **Live weather** (13 weekly aggregates from Open-Meteo) replaces every
   temperature / PM2.5 / AQI / precipitation / heat-wave feature, recomputed
   with the same rolling / lag / interaction rules as the featurize stage.
3. Missing values are filled with the training-period median, exactly as
   the train stage did before fitting.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.features import drop_first_week
from src.core.models import MODEL_LABELS
from src.utils import paths
from src.utils.config import load_params
from src.utils.io import load_model

ROOT = paths.ROOT
MODEL_DIR = paths.MODELS
FEATURED_PATH = paths.FEATURES

TRAIN_END_DATE = load_params()["split"]["train_end"]   # same chronological cut as training
COUNTRY_CODE = "BGD"

TASKS = {
    "overall":     "best_overall_classifier.joblib",
    "respiratory": "best_respiratory.joblib",
    "vector":      "best_vector.joblib",
    "heat":        "best_heat.joblib",
    "waterborne":  "best_waterborne.joblib",
    "cardio":      "best_cardio.joblib",
}

TARGETS = {
    "respiratory": "respiratory_disease_rate",
    "vector":      "vector_disease_risk_score",
    "heat":        "heat_related_admissions",
    "waterborne":  "waterborne_disease_incidents",
    "cardio":      "cardio_mortality_rate",
}


def _model_names() -> dict[str, str]:
    """Display name of each task's winner, e.g. {"overall": "Logistic Regression"}."""
    if not paths.VALIDATION_METRICS.exists():
        return {k: "–" for k in TASKS}
    winners = json.loads(paths.VALIDATION_METRICS.read_text())["winners"]
    return {k: MODEL_LABELS[winners["overall_classifier" if k == "overall" else k]] for k in TASKS}


MODEL_NAMES = _model_names()

# Weather base columns and the step-3 rules that derive features from them.
WEATHER_COLS = ["temperature_celsius", "pm25_ugm3", "air_quality_index",
                "precipitation_mm", "heat_wave_days"]
ROLL_WINDOWS = (4, 8, 12)
LAG_WEEKS = (1, 2, 4, 8)
N_WEEKS = 13                     # enough history for lag8w and roll_*_12w


def derive_weather_features(weekly: pd.DataFrame) -> dict[str, float]:
    """Step-3 rolling / lag / interaction features for the newest week.

    ``weekly`` has one row per week (oldest first) and the WEATHER_COLS.
    Rolling windows include the current week (``min_periods=1``), lags
    shift by whole weeks — both identical to the featurize stage.
    """
    out: dict[str, float] = {}
    last = len(weekly) - 1
    for col in WEATHER_COLS:
        s = weekly[col].astype(float).reset_index(drop=True)
        out[col] = float(s.iloc[last])
        for w in ROLL_WINDOWS:
            win = s.iloc[max(0, last - w + 1): last + 1]
            out[f"{col}_roll_mean_{w}w"] = float(win.mean())
            out[f"{col}_roll_sum_{w}w"] = float(win.sum())
            out[f"{col}_roll_max_{w}w"] = float(win.max())
        for lag in LAG_WEEKS:
            out[f"{col}_lag{lag}w"] = float(s.iloc[last - lag]) if last - lag >= 0 else np.nan
    out["temp_x_pm25"] = out["temperature_celsius"] * out["pm25_ugm3"]
    out["rain_x_temp"] = out["precipitation_mm"] * out["temperature_celsius"]
    out["pm25_x_aqi"] = out["pm25_ugm3"] * out["air_quality_index"]
    return out


def calendar_features(date: pd.Timestamp) -> dict[str, float]:
    week = int(date.isocalendar().week)
    return {"week": week, "month": int(date.month),
            "week_sin": float(np.sin(2 * np.pi * week / 52.0)),
            "week_cos": float(np.cos(2 * np.pi * week / 52.0))}


@dataclass
class Prediction:
    """Raw model outputs for one feature row."""
    proba: dict[str, float]          # overall classifier: {"Low","Medium","High"}
    diseases: dict[str, float]       # regressor predictions in target units


class ModelStore:
    """Loads the six models and the training reference data once."""

    def __init__(self) -> None:
        missing = [f for f in TASKS.values() if not (MODEL_DIR / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing model files in {MODEL_DIR}: {missing} "
                                    "— run `dvc pull` (or `dvc repro`).")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Single-threaded predict: bit-identical outputs, and ~2.5x faster than
            # parallel for the dashboard's small batches (thread start-up dominates).
            self.models = {k: load_model(MODEL_DIR / f) for k, f in TASKS.items()}

        self.features: list[str] = list(self.models["overall"].feature_names_in_)
        for k, m in self.models.items():
            if list(m.feature_names_in_) != self.features:
                raise RuntimeError(f"Model {k} uses a different feature list")

        # Training dropped the first week per country (NaN lag1w); mirror it.
        fd = drop_first_week(pd.read_parquet(FEATURED_PATH))
        train = fd[fd["date"] <= TRAIN_END_DATE]

        self.train_median = train[self.features].median(numeric_only=True)
        # Sorted training targets -> percentile lookup for disease scores.
        self.train_targets = {k: np.sort(train[t].dropna().to_numpy(float))
                              for k, t in TARGETS.items()}
        self.country = fd[fd[f"country_code_{COUNTRY_CODE}"] == 1].sort_values("date")
        self.template = self.country.iloc[-1]
        self.classes = [str(c) for c in self.models["overall"].classes_]

    # -- feature rows ---------------------------------------------------------
    def country_history(self, n_weeks: int) -> pd.DataFrame:
        """Last ``n_weeks`` Bangladesh rows from the dataset (oldest first)."""
        return self.country.tail(n_weeks).reset_index(drop=True)

    def build_row(self, weekly_model_scale: pd.DataFrame | None,
                  date: pd.Timestamp) -> pd.Series:
        """Template row with live weather features and today's calendar."""
        row = self.template.copy()
        if weekly_model_scale is not None:
            for k, v in derive_weather_features(weekly_model_scale).items():
                if k in row.index:
                    row[k] = v
            for k, v in calendar_features(date).items():
                if k in row.index:
                    row[k] = v
        return self.to_matrix([row]).iloc[0]

    def to_matrix(self, rows: list[pd.Series] | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
        X = df[self.features].astype(float).fillna(self.train_median)
        return X.reset_index(drop=True)

    # -- inference ------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        """Batch inference: one Prediction per row of X."""
        proba = self.models["overall"].predict_proba(X)
        dis = {k: np.asarray(self.models[k].predict(X), dtype=float) for k in TARGETS}
        return [
            Prediction(
                proba={c: float(proba[i, j]) for j, c in enumerate(self.classes)},
                diseases={k: float(dis[k][i]) for k in TARGETS},
            )
            for i in range(len(X))
        ]

    def percentile(self, key: str, value: float) -> float:
        """Percentile (0–100) of ``value`` in the training target distribution."""
        arr = self.train_targets[key]
        return float(100.0 * np.searchsorted(arr, value, side="right") / len(arr))
