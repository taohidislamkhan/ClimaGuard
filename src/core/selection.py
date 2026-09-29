"""Committee feature selection (was pipeline/step4).

1. Correlation pruning: walk features from strongest to weakest |r| with the
   target and drop any whose |r| with an already-kept feature is > threshold.
2. Three independent rankings: |Pearson r|, mutual information, Random Forest
   impurity importance.
3. Vote: a feature gets one vote per ranking where it is in the top half.
   Keep features with >= min_votes, then the top_k by mean rank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

NON_FEATURES = {"record_id", "country_name", "date", "risk_class"}


def candidate_pool(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    exclude = NON_FEATURES | {target}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].astype(np.float64), df[target].astype(np.float64)


def drop_near_duplicates(X: pd.DataFrame, y: pd.Series, threshold: float) -> tuple[list[str], list]:
    order = X.corrwith(y).abs().fillna(0.0).sort_values(ascending=False).index.tolist()
    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for col in order:
        if not kept:
            kept.append(col)
            continue
        corrs = {k: X[col].corr(X[k]) for k in kept}
        partner, raw = max(corrs.items(), key=lambda kv: abs(kv[1]) if pd.notna(kv[1]) else -1.0)
        max_val = abs(raw) if pd.notna(raw) else 0.0
        if max_val > threshold:
            dropped.append((col, partner, float(max_val)))
        else:
            kept.append(col)
    return kept, dropped


def pearson_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    return X.corrwith(y).abs().fillna(0.0)


def mi_scores(X: pd.DataFrame, y: pd.Series, sample: int, seed: int) -> pd.Series:
    Xf = X.fillna(X.median(numeric_only=True))
    if len(Xf) > sample:
        idx = np.random.default_rng(seed).choice(len(Xf), size=sample, replace=False)
        Xf, y = Xf.iloc[idx], y.iloc[idx]
    return pd.Series(mutual_info_regression(Xf, y, random_state=seed), index=X.columns)


def rf_scores(X: pd.DataFrame, y: pd.Series, trees: int, seed: int) -> pd.Series:
    Xf = X.fillna(X.median(numeric_only=True))
    rf = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1, max_features="sqrt")
    rf.fit(Xf, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="min").astype(int)


def select(df: pd.DataFrame, p: dict, seed: int) -> tuple[list[str], pd.DataFrame, dict]:
    """Returns (committee in mean-rank order, rankings table, summary)."""
    X, y = candidate_pool(df, p["target"])
    kept, dropped = drop_near_duplicates(X, y, p["corr_threshold"])
    Xk = X[kept]

    scores = {
        "pearson": pearson_scores(Xk, y),
        "mi": mi_scores(Xk, y, p["mi_sample"], seed),
        "rf": rf_scores(Xk, y, p["rf_trees"], seed),
    }
    ranks = {k: _rank(v) for k, v in scores.items()}

    n_keep_per = max(1, len(kept) // 2)
    votes = sum((ranks[k] <= n_keep_per).astype(int) for k in ranks)
    mean_rank = pd.concat(ranks.values(), axis=1).mean(axis=1)

    table = pd.DataFrame({"feature": kept})
    for k in ranks:
        table[f"{k}_score"] = scores[k].reindex(kept).values
        table[f"{k}_rank"] = ranks[k].reindex(kept).values
    table["votes"] = votes.reindex(kept).values
    table["mean_rank"] = mean_rank.reindex(kept).values

    survivors = table[table["votes"] >= p["min_votes"]].sort_values("mean_rank")
    committee = survivors.head(p["top_k"])["feature"].tolist()
    table["selected"] = table["feature"].isin(committee)
    table = table.sort_values("mean_rank").reset_index(drop=True)

    summary = {
        "candidate_features": int(X.shape[1]),
        "corr_threshold": p["corr_threshold"],
        "dropped_near_duplicates": len(dropped),
        "after_dedup": len(kept),
        "votes_needed": p["min_votes"],
        "survivors_with_min_votes": int(len(survivors)),
        "committee_size": len(committee),
        "examples_dropped": [{"drop": a, "kept": b, "abs_corr": round(c, 4)} for a, b, c in dropped[:10]],
    }
    return committee, table, summary
