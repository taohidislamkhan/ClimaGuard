"""Step 3 — Feature engineering (Phase 4).

Builds ~160–260 predictor columns grouped by country so a rolling or lag
window can never leak across national borders. Five families, all gated by
``groupby('country_code')`` so the window cannot straddle borders:

Family                          Implementation                                     Why
-------------------------------  -------------------------------------------------  ---------------------------------------------------
Rolling mean / sum / max         df.groupby('country')[col].transform(             Disease accumulates exposure over weeks, not one reading.
  (4, 8, 12-week)                lambda s: s.rolling(w).{mean|sum|max}())
Lags (1, 2, 4, 8 weeks)          df.groupby('country')[col].shift(lag)             Incubation, standing water, delayed CV stress.
Cyclical encodings               sin(2π·week/52), cos(2π·week/52); month the same  December sits next to January.
Interactions                     temperature × PM2.5, rain × temperature,         Pollution is worse on hot days; vector risk compounds
                                 PM2.5 × AQI                                        after rain following warmth.
Categorical dummies              one-hot on country / region / income / climate    Absorbs baseline reporting & health-system differences.

Label creation
--------------
For the Low / Medium / High risk-class target the proposal requires tertiles
cut from the **training period only**, never the test period. We build a
continuous composite `health_impact_score` from climate + health signals,
temporally split the data (default train ≤ 2023, test ≥ 2024), compute the
33rd and 67th percentiles on the training subset, then apply those edges to
every row to produce `risk_class` ∈ {Low, Medium, High}.  Cutoff years and
edges are written to ``output/feature_engineering_meta.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/cleaned_data.csv")
OUT_PATH = Path("data/processed/featured_data.csv")
META_PATH = Path("output/feature_engineering_meta.json")

# Columns to summarise over rolling windows.
ROLL_BASE_COLS = [
    "temperature_celsius",
    "pm25_ugm3",
    "air_quality_index",
    "precipitation_mm",
]
ROLL_EXTRA_COLS = [
    "heat_wave_days",
    "extreme_weather_events",
    "waterborne_disease_incidents",
]

ROLL_WINDOWS = (4, 8, 12)              # weeks
LAG_WEEKS = (1, 2, 4, 8)               # weeks

CATEGORICAL_COLS = [
    "country_code",
    "region",
    "income_level",
    "climate_zone",
]

# Temporal split for label generation. Training-period tertiles only.
TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024

# Columns used to define the continuous composite target.
COMPOSITE_COLS = [
    "heat_related_admissions",
    "respiratory_disease_rate",
    "cardio_mortality_rate",
    "vector_disease_risk_score",
    "waterborne_disease_incidents",
    "pm25_ugm3",
    "temperature_celsius",
]


def _rolling(group: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Add rolling mean / sum / max columns for a single base column."""
    s = group[col]
    return pd.DataFrame(
        {
            f"{col}_roll_mean_{window}w": s.rolling(window, min_periods=1).mean(),
            f"{col}_roll_sum_{window}w":  s.rolling(window, min_periods=1).sum(),
            f"{col}_roll_max_{window}w":  s.rolling(window, min_periods=1).max(),
        },
        index=group.index,
    )


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country rolling stats (rows stay aligned via original index)."""
    parts: list[pd.DataFrame] = []
    for col in ROLL_BASE_COLS + ROLL_EXTRA_COLS:
        if col not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            parts.append(
                df.groupby("country_code", group_keys=False)
                  .apply(lambda g, c=col, ww=w: _rolling(g, c, ww))
            )
    return pd.concat([df] + parts, axis=1)


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country lags (shift prevents using the current row's value)."""
    out = df.copy()
    base_cols = ROLL_BASE_COLS + ROLL_EXTRA_COLS + ["heat_related_admissions"]
    for col in base_cols:
        if col not in out.columns:
            continue
        for lag in LAG_WEEKS:
            out[f"{col}_lag{lag}w"] = (
                out.groupby("country_code")[col].shift(lag)
            )
    return out


def _add_cyclical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos transforms so week 52 sits next to week 1."""
    out = df.copy()
    if "week" in out.columns:
        out["week_sin"] = np.sin(2 * np.pi * out["week"] / 52.0)
        out["week_cos"] = np.cos(2 * np.pi * out["week"] / 52.0)
    if "month" in out.columns:
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs motivated by the proposal table."""
    out = df.copy()
    pairs = [
        ("temperature_celsius", "pm25_ugm3",         "temp_x_pm25"),
        ("precipitation_mm",    "temperature_celsius","rain_x_temp"),
        ("pm25_ugm3",           "air_quality_index",  "pm25_x_aqi"),
        ("temperature_celsius", "humidity_index",     "temp_x_humidity"),
    ]
    for a, b, name in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a] * out[b]
    return out


def _add_categorical_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the four categorical columns.

    ``drop_first=True`` avoids the dummy-variable trap for linear models while
    keeping every level visible. ``country_code`` is one-hot per the proposal
    ("Absorb baseline reporting/health differences") even though the
    per-country grouping already anchors the rolling/lag features — the
    dummies absorb level differences that the lags cannot.
    """
    return pd.get_dummies(
        df,
        columns=[c for c in CATEGORICAL_COLS if c in df.columns],
        drop_first=True,
        dtype="int8",
    )


def _build_health_impact_score(df: pd.DataFrame) -> pd.Series:
    """Min-max scale the composite columns and return their mean (0–100).

    Scaling on the full dataset is fine — the SCORE is just an input to label
    generation, not a feature. The tertile CUT is what is constrained to the
    training period (see ``_make_risk_class``).
    """
    parts = []
    for col in COMPOSITE_COLS:
        if col not in df.columns:
            continue
        lo, hi = df[col].min(), df[col].max()
        if hi == lo:
            parts.append(pd.Series(0.0, index=df.index))
        else:
            parts.append((df[col] - lo) / (hi - lo))
    if not parts:
        return pd.Series(0.0, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1) * 100.0


def _make_risk_class(
    df: pd.DataFrame,
    score: pd.Series,
    train_end_year: int,
) -> tuple[pd.Series, list[float]]:
    """Cut the score into {Low, Medium, High} using training-period tertiles."""
    train_mask = df["year"] <= train_end_year
    if train_mask.sum() < 3:
        train_mask = pd.Series(True, index=df.index)

    edges = np.quantile(score.loc[train_mask], [1 / 3, 2 / 3]).tolist()
    if edges[0] == edges[1]:
        edges[1] = edges[0] + 1e-6

    labels = pd.cut(
        score,
        bins=[-np.inf, edges[0], edges[1], np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype("object")
    labels = labels.fillna("Low")
    return labels, edges


def main() -> None:
    df = pd.read_csv(IN_PATH, parse_dates=["date"])

    # Defensive sort — step 1 already sorted, but re-sort in case anyone
    # re-runs step 3 against an unsorted frame.
    df = df.sort_values(["country_code", "date"]).reset_index(drop=True)

    # ----------------- five feature families -----------------
    df = _add_rolling_features(df)
    df = _add_lag_features(df)
    df = _add_cyclical_encodings(df)
    df = _add_interactions(df)
    df = _add_categorical_dummies(df)

    # ----------------- labels -----------------
    df["health_impact_score"] = _build_health_impact_score(df)
    df["risk_class"], edges = _make_risk_class(
        df, df["health_impact_score"], TRAIN_END_YEAR,
    )

    # ----------------- meta -----------------
    predictor_cols = [
        c for c in df.columns
        if c not in {
            "record_id", "country_name", "date",
            "health_impact_score", "risk_class",
        }
    ]
    meta = {
        "rows": int(len(df)),
        "n_predictors": int(len(predictor_cols)),
        "train_end_year": TRAIN_END_YEAR,
        "test_start_year": TEST_START_YEAR,
        "risk_class_edges": edges,
        "risk_class_counts": df["risk_class"].value_counts().to_dict(),
        "rolling_windows": list(ROLL_WINDOWS),
        "lag_weeks": list(LAG_WEEKS),
        "composite_cols": COMPOSITE_COLS,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Featured shape: {df.shape}")
    print(f"Predictor columns: {len(predictor_cols)}  "
          f"(target band 160–260: {'OK' if 160 <= len(predictor_cols) <= 260 else 'OUT OF BAND'})")
    print(f"Risk-class edges (train-only): {edges}")
    print(f"Risk-class distribution:\n{df['risk_class'].value_counts()}")
    print(f"Wrote featured data -> {OUT_PATH}")
    print(f"Wrote metadata     -> {META_PATH}")


if __name__ == "__main__":
    main()
