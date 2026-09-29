"""Feature engineering and label creation (was pipeline/step3).

Every rolling window and lag is computed inside ``groupby('country_code')``,
so a window never crosses from one country into the next.

Note: rolling windows use ``min_periods=1`` and include the current week.
For ``waterborne_disease_incidents`` this means the waterborne regressor sees
the current week's value through its rolling features (target leakage). This
is kept as in the original analysis and documented in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG1_SUFFIX = "_lag1w"


def _rolling(group: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    s = group[col]
    return pd.DataFrame(
        {
            f"{col}_roll_mean_{window}w": s.rolling(window, min_periods=1).mean(),
            f"{col}_roll_sum_{window}w":  s.rolling(window, min_periods=1).sum(),
            f"{col}_roll_max_{window}w":  s.rolling(window, min_periods=1).max(),
        },
        index=group.index,
    )


def add_rolling_features(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    parts = []
    for col in cols:
        for w in windows:
            parts.append(df.groupby("country_code", group_keys=False)
                           .apply(lambda g, c=col, ww=w: _rolling(g, c, ww)))
    return pd.concat([df] + parts, axis=1)


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        for lag in lags:
            out[f"{col}_lag{lag}w"] = out.groupby("country_code")[col].shift(lag)
    return out


def add_cyclical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos so week 52 sits next to week 1 and December next to January."""
    out = df.copy()
    out["week_sin"] = np.sin(2 * np.pi * out["week"] / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * out["week"] / 52.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def add_interactions(df: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
    out = df.copy()
    for a, b, name in pairs:
        out[name] = out[a] * out[b]
    return out


def add_categorical_dummies(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot with ``drop_first`` (avoids the dummy trap for the linear models)."""
    return pd.get_dummies(df, columns=cols, drop_first=True, dtype="int8")


def health_impact_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Mean of min-max scaled columns, 0-100. Only used to build the label, never as a feature."""
    parts = []
    for col in cols:
        lo, hi = df[col].min(), df[col].max()
        parts.append(pd.Series(0.0, index=df.index) if hi == lo else (df[col] - lo) / (hi - lo))
    return pd.concat(parts, axis=1).mean(axis=1) * 100.0


def risk_class(df: pd.DataFrame, score: pd.Series, train_end_year: int) -> tuple[pd.Series, list[float]]:
    """Low / Medium / High from tertiles of the score, cut on training years only."""
    train_mask = df["year"] <= train_end_year
    edges = np.quantile(score.loc[train_mask], [1 / 3, 2 / 3]).tolist()
    if edges[0] == edges[1]:
        edges[1] = edges[0] + 1e-6
    labels = pd.cut(score, bins=[-np.inf, edges[0], edges[1], np.inf],
                    labels=["Low", "Medium", "High"], include_lowest=True).astype("object")
    return labels.fillna("Low"), edges


def build_features(df: pd.DataFrame, fp: dict, lp: dict) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values(["country_code", "date"]).reset_index(drop=True)
    df = add_rolling_features(df, fp["roll_cols"], fp["rolling_windows"])
    df = add_lag_features(df, fp["lag_cols"], fp["lags"])
    df = add_cyclical_encodings(df)
    df = add_interactions(df, fp["interactions"])
    df = add_categorical_dummies(df, fp["categorical"])

    df["health_impact_score"] = health_impact_score(df, lp["composite_cols"])
    df["risk_class"], edges = risk_class(df, df["health_impact_score"], lp["train_end_year"])

    predictors = [c for c in df.columns
                  if c not in {"record_id", "country_name", "date", "health_impact_score", "risk_class"}]
    summary = {
        "rows": int(len(df)),
        "n_predictors": len(predictors),
        "risk_class_edges": edges,
        "risk_class_counts": df["risk_class"].value_counts().to_dict(),
    }
    return df, summary


def drop_first_week(df: pd.DataFrame) -> pd.DataFrame:
    """Drop each country's first week, where every lag-1 feature is NaN."""
    lag1 = [c for c in df.columns if c.endswith(LAG1_SUFFIX)][0]
    return df[df[lag1].notna()].reset_index(drop=True)
