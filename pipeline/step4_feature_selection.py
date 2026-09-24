"""Step 4 — Feature selection (Phase 5).

Pipeline:

1. **Build candidate pool.** Numeric predictors only, with the first week
   per country dropped (every lag feature is NaN there — see step 3).
2. **Drop near-duplicates.** Greedy filter: walk candidates in order of
   strongest correlation with the target, drop any feature whose pairwise
   correlation with an already-kept feature exceeds ``CORR_DROP``. This
   keeps the one with the stronger signal rather than blindly dropping by
   column name (PM2.5 vs AQI will be handled here, data-driven).
3. **Three independent rankings** of the surviving candidates:
   - **Pearson** ``|r|`` with the target (linear dependence)
   - **Mutual information** (any dependence, non-parametric)
   - **Tree-based importance / gain** from a small RandomForest (non-linear
     + interactions)
4. **Committee vote.** Keep features that surface in **≥ 2 of 3** rankings
   (``MIN_VOTES = 2``). Then take the top ``TOP_K = 60`` by mean rank for
   a committee-readable feature set.
5. **Save two artefacts:**
   - ``feature_selection.csv`` — every surviving candidate with its rank in
     each of the three rankings side by side. This is the Slide 5 evidence
     ("Proposed Feature Importance & Selection techniques").
   - ``selected_features.csv`` — just the chosen feature columns plus the
     target, ready for step 5.

O(p) tricks: correlation with the target uses ``corrwith``; the greedy
near-duplicate pass builds each feature's correlation vector on demand
rather than materialising a full p×p matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

IN_PATH = Path("data/processed/featured_data.csv")
RANKINGS_OUT = Path("data/processed/feature_selection.csv")
SELECTED_OUT = Path("data/processed/selected_features.csv")
META_PATH = Path("output/feature_selection_meta.json")

TARGET = "health_impact_score"
RISK_CLASS = "risk_class"

CORR_DROP = 0.95                 # drop a feature whose |r| with a kept feature > this
MIN_VOTES = 2                    # a feature must appear in ≥ MIN_VOTES rankings to survive
TOP_K = 60                       # cap the committee output at the top-K by mean rank
MI_SAMPLE = 4000                 # subsample size for MI (full n is 14k and MI is O(n·p))
RF_TREES = 200
RF_SEED = 42
MI_SEED = 42


def _drop_first_week_per_country(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where any lag-1w column is NaN (= first week per country)."""
    lag_cols = [c for c in df.columns if c.endswith("_lag1w")]
    if not lag_cols:
        return df, 0
    mask = df[lag_cols[0]].isna()
    return df.loc[~mask].reset_index(drop=True), int(mask.sum())


def _build_candidate_pool(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X numeric, y target, candidate column names)."""
    exclude = {target, RISK_CLASS, "record_id", "country_name", "date", "risk_class"}
    candidates = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[candidates].astype(np.float64)
    y = df[target].astype(np.float64)
    return X, y, candidates


def _drop_near_duplicates(
    X: pd.DataFrame, y: pd.Series, threshold: float
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy: keep the feature with the strongest |r| to target first.

    For each subsequent candidate (in target-correlation order), if its
    |corr| with any already-kept feature exceeds ``threshold``, drop it.
    Returns the kept column names and the dropped pairs (with their
    correlation) for the meta file.
    """
    target_corr = X.corrwith(y).abs().fillna(0.0)
    order = target_corr.sort_values(ascending=False).index.tolist()

    kept: list[str] = []
    kept_vecs: list[pd.Series] = []
    dropped_pairs: list[tuple[str, str, float]] = []

    for col in order:
        if len(kept_vecs) == 0:
            kept.append(col)
            kept_vecs.append(X[col])
            continue

        # Max |corr| with everything already kept. Compute each pairwise
        # corr directly and take the max; avoids an axis ambiguity in
        # pd.concat over differently-indexed 1-element Series.
        corrs = {v.name: X[col].corr(v) for v in kept_vecs}
        partner, raw = max(corrs.items(), key=lambda kv: abs(kv[1]) if pd.notna(kv[1]) else -1.0)
        max_val = abs(raw) if pd.notna(raw) else 0.0
        if max_val > threshold:
            dropped_pairs.append((col, partner, float(max_val)))
            continue
        kept.append(col)
        kept_vecs.append(X[col])

    return kept, dropped_pairs


def _pearson_ranks(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """|Pearson r| with the target, then rank ascending (rank 1 = best)."""
    r = X.corrwith(y).abs().fillna(0.0)
    return r.rank(ascending=False, method="min").astype(int)


def _tree_ranks(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Impurity-based importance from a small RF, ranked ascending.

    sklearn 1.6+ rejects NaN inputs. Fill lag NaNs with the per-column
    median before fitting so the forest sees the same data shape as MI.
    """
    Xf = X.fillna(X.median(numeric_only=True))
    rf = RandomForestRegressor(
        n_estimators=RF_TREES,
        random_state=RF_SEED,
        n_jobs=1,            # n_jobs=1 for bit-reproducible runs
        max_features="sqrt",
    )
    rf.fit(Xf, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    return imp.rank(ascending=False, method="min").astype(int)


def _mi_ranks(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mutual information (any dependence), ranked ascending.

    MI requires no NaN. After step 1 drops the first week per country only
    lag-1w is NaN-free; lag 2/4/8 still have NaN at the first 2/4/8 weeks.
    We fill those with the per-column median (a neutral imputation that
    does not invent a signal). MI is O(n·p); we subsample to keep this
    snappy on 14k rows × ~150 features.
    """
    Xf = X.fillna(X.median(numeric_only=True))
    if len(Xf) > MI_SAMPLE:
        rng = np.random.default_rng(MI_SEED)
        idx = rng.choice(len(Xf), size=MI_SAMPLE, replace=False)
        Xs, ys = Xf.iloc[idx], y.iloc[idx]
    else:
        Xs, ys = Xf, y
    mi = mutual_info_regression(Xs, ys, random_state=MI_SEED)
    s = pd.Series(mi, index=X.columns)
    return s.rank(ascending=False, method="min").astype(int)


def main() -> None:
    df = pd.read_csv(IN_PATH)
    if TARGET not in df.columns:
        print(f"Target '{TARGET}' not found; skipping selection.")
        return

    print(f"Input shape: {df.shape}")
    df, dropped_rows = _drop_first_week_per_country(df)
    if dropped_rows:
        print(f"Dropped {dropped_rows} first-week-per-country rows (NaN lags)")

    X, y, candidates = _build_candidate_pool(df, TARGET)
    print(f"Candidate features: {len(candidates)}")

    # ---------- 1. near-duplicate filter ----------
    kept, dropped_pairs = _drop_near_duplicates(X, y, CORR_DROP)
    print(f"After |r|>{CORR_DROP} near-duplicate filter: {len(kept)} kept "
          f"({len(dropped_pairs)} dropped)")
    Xk = X[kept]

    # ---------- 2. three independent rankings ----------
    pearson_rank = _pearson_ranks(Xk, y)
    mi_rank      = _mi_ranks(Xk, y)
    tree_rank    = _tree_ranks(Xk, y)

    # Convert ranks to a "vote": a feature is "in the ranking" if its rank
    # is ≤ n_keep_per_ranking. We let each ranking cast a vote for the top
    # half of its features, so the committee has a stable size regardless
    # of where the long tail sits.
    n_keep_per = max(1, len(kept) // 2)
    in_pearson = set(pearson_rank[pearson_rank <= n_keep_per].index)
    in_mi      = set(mi_rank[mi_rank <= n_keep_per].index)
    in_tree    = set(tree_rank[tree_rank <= n_keep_per].index)

    votes = pd.DataFrame({
        "pearson": [int(c in in_pearson) for c in pearson_rank.index],
        "mi":      [int(c in in_mi)      for c in mi_rank.index],
        "tree":    [int(c in in_tree)    for c in tree_rank.index],
    }, index=pearson_rank.index)
    votes["vote_count"] = votes.sum(axis=1)
    votes["mean_rank"]  = pd.concat([pearson_rank, mi_rank, tree_rank], axis=1).mean(axis=1)

    survivors = votes[votes["vote_count"] >= MIN_VOTES].sort_values("mean_rank")
    committee = survivors.head(TOP_K).index.tolist()
    print(f"Votes: >={MIN_VOTES} rankings -> {len(survivors)}; "
          f"top-{TOP_K} by mean rank -> {len(committee)}")

    # ---------- 3. save rankings side by side ----------
    rankings = pd.DataFrame({
        "feature":       kept,
        "pearson_rank":  pearson_rank.reindex(kept).values,
        "mi_rank":       mi_rank.reindex(kept).values,
        "tree_rank":     tree_rank.reindex(kept).values,
        "vote_count":    votes["vote_count"].reindex(kept).values,
        "mean_rank":     votes["mean_rank"].reindex(kept).values,
        "kept":          [c in committee for c in kept],
    }).sort_values("mean_rank").reset_index(drop=True)
    RANKINGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(RANKINGS_OUT, index=False)

    # ---------- 4. save selected features for step 5 ----------
    out_cols = committee + [TARGET]
    if RISK_CLASS in df.columns:
        out_cols.append(RISK_CLASS)
    SELECTED_OUT.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(SELECTED_OUT, index=False)

    # ---------- 5. meta ----------
    meta = {
        "input_rows": int(len(df) + dropped_rows),
        "input_cols": int(df.shape[1]),
        "dropped_first_week_rows": dropped_rows,
        "candidate_features": len(candidates),
        "corr_drop_threshold": CORR_DROP,
        "n_dropped_near_duplicates": len(dropped_pairs),
        "examples_dropped_pairs": [
            {"drop": a, "kept": b, "abs_corr": round(c, 4)}
            for a, b, c in dropped_pairs[:10]
        ],
        "after_dedup": len(kept),
        "n_keep_per_ranking": n_keep_per,
        "min_votes": MIN_VOTES,
        "top_k": TOP_K,
        "survivors_with_min_votes": int(len(survivors)),
        "committee_size": len(committee),
        "committee": committee,
        "top10_by_mean_rank": survivors.head(10).reset_index().rename(
            columns={"index": "feature"}
        ).to_dict(orient="records"),
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2, default=str))

    print(f"Wrote rankings side-by-side -> {RANKINGS_OUT}")
    print(f"Wrote selected features       -> {SELECTED_OUT}")
    print(f"Wrote metadata                -> {META_PATH}")


if __name__ == "__main__":
    main()
