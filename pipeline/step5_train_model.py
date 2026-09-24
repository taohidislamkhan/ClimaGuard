"""Step 5 — Phase 6 modeling.

Train one **overall** risk classifier + **five disease-specific regressors**
on the 54-feature committee selected in step 4. Chronological split is the
single most important methodological choice (random splits leak future
weeks into the training set via the rolling/lag columns):

    Train: 2015-01-04 -> 2022-12-31
    Val  : 2023-01-01 -> 2023-12-31   (used only to pick the best model)
    Test : 2024-01-01 -> 2025-10-19   (never touched for tuning or cutoffs)

Tasks
-----
overall_classifier   risk_class (Low / Medium / High)          -- 4 models
respiratory          respiratory_disease_rate                  -- 4 models
cardio               cardio_mortality_rate                     -- 4 models
vector               vector_disease_risk_score                 -- 4 models
waterborne           waterborne_disease_incidents              -- 4 models
heat                 heat_related_admissions                   -- 4 models
=> 24 fits total.

Models
------
Logistic Regression (elastic-net) -- baseline, linear
Decision Tree                    -- interpretability baseline
Random Forest                    -- bagging ensemble
XGBoost                          -- boosting ensemble (skipped if not installed)

Evaluation
----------
Classifier: accuracy, macro-F1, ROC-AUC (one-vs-rest, weighted).
Regressor : RMSE, MAE, R^2.

Selection (defensible)
----------------------
- Classifier  : highest validation macro-F1.
- Regressor   : lowest validation RMSE.
The TEST split is computed for every model and reported in the metrics
table, but is **not** used for selection — that is what makes the test
numbers honest.

Artefacts
---------
models/phase6/best_<task>.joblib        (6 files)
output/phase6_metrics.csv               (all 24 models x 3 splits = 72 rows)
output/phase6_run_summary.json          (split sizes, runtime, best-model map)

Notes
-----
- ``selected_features.csv`` has no ``date`` or ``record_id``; we re-derive
  those columns by position-aligning against ``featured_data.csv`` (both
  files share the row order produced by step 4's NaN-row drop).
- ``cardio_mortality_rate`` is not in the selected features (it was filtered
  out by the dedup / committee vote); we left-join it back from
  ``featured_data.csv`` for the cardio regressor only.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAVE_XGB = True
except Exception:
    XGBClassifier = XGBRegressor = None  # type: ignore
    _HAVE_XGB = False

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
FEATURES_PATH = Path("data/processed/featured_data.csv")
SELECTED_PATH = Path("data/processed/selected_features.csv")
MODEL_DIR     = Path("models/phase6")
METRICS_PATH  = Path("output/phase6_metrics.csv")
SUMMARY_PATH  = Path("output/phase6_run_summary.json")

TARGET_LABEL = "risk_class"
COMPOSITE_TARGET = "health_impact_score"  # derived from the 5 diseases — leak risk if kept as a feature
RISK_TARGETS = [TARGET_LABEL]

DISEASE_TARGETS = [
    "respiratory_disease_rate",
    "cardio_mortality_rate",
    "vector_disease_risk_score",
    "waterborne_disease_incidents",
    "heat_related_admissions",
]

ALL_TASKS = [
    ("overall_classifier", "classification", TARGET_LABEL),
    ("respiratory",        "regression",     "respiratory_disease_rate"),
    ("cardio",             "regression",     "cardio_mortality_rate"),
    ("vector",             "regression",     "vector_disease_risk_score"),
    ("waterborne",         "regression",     "waterborne_disease_incidents"),
    ("heat",               "regression",     "heat_related_admissions"),
]

TRAIN_END_DATE  = "2022-12-31"
VAL_END_DATE    = "2023-12-31"
RANDOM_STATE    = 42

# sklearn 1.8 deprecates ``penalty='elasticnet'`` on LogisticRegression; the
# keyword is still honoured but emits a FutureWarning every call. The new
# style uses ``l1_ratio`` alone with default penalty, but only with the
# saga solver. Keep the warning quiet until we migrate.
warnings.filterwarnings(
    "ignore",
    message=r".*penalty.*elasticnet.*",
    category=FutureWarning,
)


# ---------------------------------------------------------------------------
# Determinism: pin Python + NumPy global state. Every model factory below
# already passes ``random_state=RANDOM_STATE``; this catches stray helpers
# that might call ``np.random.*`` without explicit seeding.
# ---------------------------------------------------------------------------
def _seed_everything(seed: int = RANDOM_STATE) -> None:
    import os, random as _random
    _random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_seed_everything()


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load selected features and align with date / cardio columns.

    Returns
    -------
    df        : selected features + aligned ``date``, ``year``, ``cardio_mortality_rate``
    y_frame   : classification & regression target columns aligned to df
    feat_cols : numeric predictor column names (excluding any target)
    """
    sf = pd.read_csv(SELECTED_PATH)
    fd = pd.read_csv(
        FEATURES_PATH,
        usecols=["date", "year", "cardio_mortality_rate"],
    )

    # Position-align: step 4 dropped the 25 rows where the lag1w columns
    # were NaN, so featured[i] and selected[i] match iff i is past the
    # cumulative drop for that country. The easiest reliable alignment is
    # to recompute the drop mask and reindex.
    fd_full = pd.read_csv(FEATURES_PATH)
    lag1_cols = [c for c in fd_full.columns if c.endswith("_lag1w")]
    keep_mask = fd_full[lag1_cols[0]].notna()
    fd_aligned = fd.loc[keep_mask].reset_index(drop=True)

    if len(fd_aligned) != len(sf):
        raise RuntimeError(
            f"Alignment check failed: featured(kept)={len(fd_aligned)} "
            f"vs selected={len(sf)}. Refusing to proceed."
        )

    df = sf.copy()
    df["date"] = pd.to_datetime(fd_aligned["date"])
    df["year"] = fd_aligned["year"].astype(int)
    df["cardio_mortality_rate"] = fd_aligned["cardio_mortality_rate"].values

    # Targets live in their own frame so we don't have to remember which
    # columns were also accidentally included as predictors. Note:
    # ``health_impact_score`` is the composite the dataset's own step 3
    # builds from the five disease targets + climate — including it as a
    # predictor would let every regressor leak the answer through a single
    # column. We always exclude it from the predictor set, even for the
    # classifier (the classifier's target is ``risk_class``, but
    # ``health_impact_score`` is what defined ``risk_class``'s tertiles).
    target_cols = [TARGET_LABEL, COMPOSITE_TARGET] + DISEASE_TARGETS
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Targets missing from selected_features.csv: {missing}")
    y_frame = df[target_cols].copy()

    # Master predictor matrix: everything numeric except the targets.
    # Per-task adjustments inside the training loop further drop the
    # other disease columns (so a regressor for ``vector_disease_risk``
    # does not see ``respiratory_disease_rate`` as a predictor) and drop
    # the task's own target — both forms of leakage that would inflate R².
    drop_from_X = set(target_cols) | {"date", "year"}
    feat_cols = [c for c in df.columns
                 if c not in drop_from_X and pd.api.types.is_numeric_dtype(df[c])]

    return df[feat_cols + ["date", "year"]], y_frame, feat_cols


def _chronological_splits(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks for train / val / test by date cutoff."""
    dates = pd.to_datetime(df["date"])
    return {
        "train": (dates <= TRAIN_END_DATE).to_numpy(),
        "val":   ((dates > TRAIN_END_DATE) & (dates <= VAL_END_DATE)).to_numpy(),
        "test":  (dates > VAL_END_DATE).to_numpy(),
    }


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def _classifiers() -> dict[str, object]:
    """The four required classifiers."""
    out: dict[str, object] = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        # XGB classifier is created lazily inside main() once we know the
        # class labels (it needs integer-encoded y).
        "dt": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE,
        ),
    }
    if _HAVE_XGB:
        out["xgb"] = None  # built per-task with the right num_class / label mapping
    return out


def _regressors() -> dict[str, object]:
    """The four required regressors.

    Logistic-regression's regression analogue is ElasticNet (linear with
    L1+L2). Standardised inputs are not strictly required for tree models
    or XGBoost, so we feed the raw matrix to those and the standardised
    matrix only to ElasticNet via a Pipeline.
    """
    out: dict[str, object] = {
        "elasticnet": Pipeline([
            ("scale", StandardScaler()),
            ("reg", ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000,
                               random_state=RANDOM_STATE)),
        ]),
        "dt": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE,
        ),
        "rf": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE,
        ),
    }
    if _HAVE_XGB:
        out["xgb"] = None  # built per-task so we can fix num_class
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _metrics_classifier(y_true, y_pred, y_proba) -> dict[str, float]:
    """Accuracy, macro-F1, OvR-weighted ROC-AUC. NaN-safe."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc_ovr_weighted": float("nan"),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
        except (ValueError, TypeError):
            # Fall back for sklearn 1.9 where multi_class arg was removed.
            try:
                out["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true, y_proba, average="weighted")
                )
            except Exception:
                pass
    return out


def _metrics_regressor(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true=y_true, y_pred=y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def _safe_predict_proba(model, X) -> np.ndarray | None:
    """predict_proba if the model supports it, else None. Swallows exceptions."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()
    print(f"[step5] loading data ...", flush=True)
    X_full, y_frame, feat_cols = _load_data()
    splits = _chronological_splits(X_full)

    n_train, n_val, n_test = int(splits["train"].sum()), int(splits["val"].sum()), int(splits["test"].sum())
    print(f"[step5] features={len(feat_cols)}  "
          f"train={n_train}  val={n_val}  test={n_test}", flush=True)
    if n_val == 0 or n_test == 0:
        raise RuntimeError("Empty val or test split — check date cutoffs.")

    # For Logistic Regression / ElasticNet we median-impute NaNs; trees
    # and XGBoost handle NaN natively.
    train_pos = np.flatnonzero(splits["train"])
    val_pos   = np.flatnonzero(splits["val"])
    test_pos  = np.flatnonzero(splits["test"])

    # Median-imputation statistics come from the training rows only so
    # val and test never leak information into the imputes. Per-task
    # feature subsets are built inside the loop because some disease
    # targets appear as predictors (the committee kept them) and must be
    # dropped to avoid trivial self-prediction.
    train_median = X_full.iloc[train_pos][feat_cols].median(numeric_only=True)

    rows: list[dict] = []
    best_per_task: dict[str, tuple[str, float]] = {}
    fitted: dict[str, dict[str, object]] = {}

    for task_name, task_kind, target in ALL_TASKS:
        print(f"\n[step5] === task: {task_name} (target={target}, kind={task_kind}) ===", flush=True)

        # Leak-free feature subset.
        # - Always drop the task's own target (would be trivial self-fit).
        # - For a disease-specific regressor, also drop the *other* disease
        #   targets: the brief asks for "one model per disease because a
        #   single generic model averages away the fact that PM2.5 drives
        #   respiratory while temperature drives vector". Leaving other
        #   diseases in the predictor set would inflate R² via cross-
        #   disease confounding, which is the opposite of the brief's
        #   intent. The overall classifier keeps everything (it is a meta
        #   task that legitimately uses the disease columns).
        if task_kind == "regression":
            drop_diseases = set(DISEASE_TARGETS)
        else:
            drop_diseases = set()
        task_feat_cols = [c for c in feat_cols if c != target and c not in drop_diseases]

        # Build y for this task, dropping rows with missing target values.
        y_full = y_frame[target].to_numpy()
        ok = ~pd.isna(y_full)

        train_idx = train_pos[ok[train_pos]]
        val_idx   = val_pos[ok[val_pos]]
        test_idx  = test_pos[ok[test_pos]]

        if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
            print(f"  skipping {task_name}: empty split after NaN filter "
                  f"(train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")
            continue

        # X_full is position-aligned with itself, so indexing by absolute
        # position works regardless of the median-fill earlier.
        Xtr = X_full.iloc[train_idx][task_feat_cols].fillna(train_median).reset_index(drop=True)
        Xva = X_full.iloc[val_idx][task_feat_cols].fillna(train_median).reset_index(drop=True)
        Xte = X_full.iloc[test_idx][task_feat_cols].fillna(train_median).reset_index(drop=True)

        y_train = y_full[train_idx]
        y_val   = y_full[val_idx]
        y_test  = y_full[test_idx]

        dropped = sorted(set(feat_cols) - set(task_feat_cols))
        print(f"  features used: {len(task_feat_cols)} "
              f"(dropped {len(dropped)}: {', '.join(dropped[:6])}{'...' if len(dropped) > 6 else ''})",
              flush=True)

        models = _classifiers() if task_kind == "classification" else _regressors()

        # XGBoost needs integer-encoded y for classification and an
        # explicit num_class. Wire it up here with the actual class set.
        if _HAVE_XGB and "xgb" in models and models["xgb"] is None:
            if task_kind == "classification":
                classes = sorted(np.unique(y_train))
                if not np.issubdtype(np.array(classes).dtype, np.integer):
                    cls_to_int = {c: i for i, c in enumerate(classes)}
                    y_train_e = np.array([cls_to_int[v] for v in y_train])
                    y_val_e   = np.array([cls_to_int[v] for v in y_val])
                    y_test_e  = np.array([cls_to_int[v] for v in y_test])
                else:
                    cls_to_int = {c: c for c in classes}
                    y_train_e, y_val_e, y_test_e = y_train, y_val, y_test
                models["xgb"] = XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(classes),
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    tree_method="hist",
                    n_jobs=1,            # bit-reproducible runs
                    random_state=RANDOM_STATE,
                    eval_metric="mlogloss",
                )
            else:
                models["xgb"] = XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    tree_method="hist",
                    n_jobs=1,            # bit-reproducible runs
                    random_state=RANDOM_STATE,
                )
        fitted[task_name] = {}

        for model_name, model in models.items():
            t_fit = time.perf_counter()

            # For XGBoost classifier we need integer y; pass the encoded
            # arrays and remember how to decode predictions back.
            int_to_cls: dict[int, object] | None = None
            if model_name == "xgb" and task_kind == "classification":
                model.fit(Xtr, y_train_e)
                int_to_cls = {i: c for c, i in cls_to_int.items()}
            else:
                model.fit(Xtr, y_train)

            fit_seconds = time.perf_counter() - t_fit

            preds_val_raw  = model.predict(Xva)
            preds_test_raw = model.predict(Xte)

            if task_kind == "classification":
                proba_val  = _safe_predict_proba(model, Xva)
                proba_test = _safe_predict_proba(model, Xte)
            else:
                proba_val = proba_test = None

            # Decode XGB classifier outputs back to original class labels.
            if model_name == "xgb" and task_kind == "classification" and int_to_cls is not None:
                preds_val  = np.array([int_to_cls[int(v)] for v in preds_val_raw])
                preds_test = np.array([int_to_cls[int(v)] for v in preds_test_raw])
            else:
                preds_val, preds_test = preds_val_raw, preds_test_raw

            if task_kind == "classification":
                m_val  = _metrics_classifier(y_val,  preds_val,  proba_val)
                m_test = _metrics_classifier(y_test, preds_test, proba_test)
            else:
                m_val  = _metrics_regressor(y_val,  preds_val)
                m_test = _metrics_regressor(y_test, preds_test)

            print(f"  {model_name:11s} fit={fit_seconds:5.1f}s  "
                  f"val={ {k: round(v, 4) for k, v in m_val.items()} }", flush=True)

            for split_name, m in (("val", m_val), ("test", m_test)):
                row = {
                    "task":   task_name,
                    "target": target,
                    "model":  model_name,
                    "split":  split_name,
                    "n_rows": int(len(y_val) if split_name == "val" else len(y_test)),
                    **m,
                }
                rows.append(row)

            fitted[task_name][model_name] = model

        # Pick best model on the val split using the right primary metric.
        val_rows = [r for r in rows if r["task"] == task_name and r["split"] == "val"]
        if task_kind == "classification":
            val_rows.sort(key=lambda r: (-r["macro_f1"], -r["accuracy"]))
        else:
            val_rows.sort(key=lambda r: (r["rmse"], r["mae"]))
        best_name, best_metric = val_rows[0]["model"], (
            val_rows[0]["macro_f1"] if task_kind == "classification" else val_rows[0]["rmse"]
        )
        best_per_task[task_name] = (best_name, best_metric)
        primary = "val_macro_f1" if task_kind == "classification" else "val_rmse"
        print(f"  best by {primary}: {best_name} ({best_metric:.4f})", flush=True)

    # ---------------------------------------------------------------- save
    metrics_df = pd.DataFrame(rows)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for task_name, (best_name, _) in best_per_task.items():
        joblib.dump(
            fitted[task_name][best_name],
            MODEL_DIR / f"best_{task_name}.joblib",
        )

    # Mark the winning row in a wide summary table (best by val metric).
    summary = {
        "xgb_available": _HAVE_XGB,
        "n_features": len(feat_cols),
        "split_sizes": {"train": n_train, "val": n_val, "test": n_test},
        "primary_metric_by_kind": {
            "classification": "val_macro_f1 (higher is better)",
            "regression":     "val_rmse (lower is better)",
        },
        "best_per_task": {
            task: {"model": name,
                   "metric": float(metric)}
            for task, (name, metric) in best_per_task.items()
        },
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    # Console summary
    print(f"\n[step5] metrics table  -> {METRICS_PATH}", flush=True)
    print(f"[step5] best models     -> {MODEL_DIR}", flush=True)
    print(f"[step5] run summary     -> {SUMMARY_PATH}", flush=True)
    print(f"[step5] total runtime   {time.perf_counter() - t0:.1f}s", flush=True)
    print("\nBest model per task (selected on validation split):")
    for task, (name, m) in best_per_task.items():
        print(f"  {task:20s} -> {name:10s}  metric={m:.4f}")


if __name__ == "__main__":
    main()
