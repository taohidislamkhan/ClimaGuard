"""Step 6 — Phase 8: Health Advisory Rule Engine.

Translates the Phase 6 model predictions into concrete public-health
action text, applied to the held-out test split only (same chronological
window as Phase 6/7: weeks from 2024-01-01 onward).

Design
------
Each test row gets:
  1. A **predicted risk class** (``Low``/``Medium``/``High``) from the
     overall classifier — by argmax over its class probabilities.
  2. Per-disease **predicted scores** from the five disease regressors.
  3. **Trigger checks** at configurable thresholds:
       - ``class_high_prob``  : classifier's P(High) threshold
                                 (default 0.70 — the lower end of the
                                  brief's 0.7–0.8 band).
       - per-disease thresholds expressed as **percentiles of the
         training-period target distribution** (e.g. 0.80 = top quintile).
         The training period (≤ 2022-12-31) is used to compute cutoffs so
         the test set never defines what counts as "elevated".
  4. An **advisory string** composed of any triggered rules. Rules are
     joined with a header line ("RISK: Low/Medium/High") plus a body that
     lists the concrete actions.

Thresholds live in one dict at the top of the file (``THRESHOLDS``) so a
tuner can tweak them without code changes. The 0.7 / 0.8 numbers from the
brief are the defaults; every entry is documented.

Inputs / outputs
----------------
IN : models/phase6/best_<task>.joblib (Phase 6 winners)
     data/processed/featured_data.csv (re-derived for date + train stats)
OUT: output/advisories.csv
     output/advisory_run_summary.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Reuse the exact loaders / split from step 5 so feature alignment is
# guaranteed to match what the models were trained on.
import os, sys
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
from step5_train_model import (  # noqa: E402
    _load_data,
    _chronological_splits,
    _seed_everything,
    DISEASE_TARGETS,
    TARGET_LABEL,
    RANDOM_STATE,
)

_seed_everything(RANDOM_STATE)

MODEL_DIR     = Path("models/phase6")
FEATURES_PATH = Path("data/processed/featured_data.csv")
OUT_CSV       = Path("output/advisories.csv")
OUT_SUMMARY   = Path("output/advisory_run_summary.json")

# ---------------------------------------------------------------------------
# Advisory catalogue (the brief's REQUIRED messages, plus a Medium layer for
# the overall risk class — public-health comms need both trigger AND severity)
# ---------------------------------------------------------------------------
ADVISORY_RULES: dict[str, str] = {
    # Triggered when the corresponding regressor's prediction clears its
    # threshold (see THRESHOLDS below).
    "respiratory_high": (
        "Respiratory risk elevated: wear a mask outdoors, "
        "use air purifiers indoors, avoid outdoor exercise during peak PM2.5 hours."
    ),
    "vector_high": (
        "Vector-borne disease risk elevated: drain stagnant water, "
        "use mosquito nets and repellents, wear long sleeves at dusk."
    ),
    "waterborne_high": (
        "Waterborne risk elevated: boil or treat drinking water, "
        "avoid contact with floodwater, wash hands frequently."
    ),
    "heat_high": (
        "Heat-related risk elevated: stay hydrated, avoid midday exertion, "
        "check on elderly and vulnerable individuals."
    ),
    # Overall-classifier severity layers (Low/Medium/High), driven by the
    # predicted class argmax, not a threshold.
    "overall_medium": (
        "Moderate overall health risk: monitor local advisories, "
        "prepare cooling / clean-air supplies."
    ),
    "overall_high": (
        "High overall health risk: activate emergency public-health protocols, "
        "scale clinic capacity, broadcast protective-behaviour guidance."
    ),
}

# ---------------------------------------------------------------------------
# Configurable thresholds. Defaults sit at the lower end of the brief's
# 0.7–0.8 band so the engine fires with reasonable sensitivity; raise
# them to reduce false alarms during testing.
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    # Classifier: probability that the predicted class is "High" must
    # exceed this for a high-overall trigger.
    "class_high_prob":      0.70,
    # Per-disease regressor triggers: quantile of the TRAINING target
    # distribution above which we treat the prediction as elevated.
    # 0.80 = top quintile, which is a standard public-health alert band.
    "respiratory_q":        0.80,
    "vector_q":             0.80,
    "waterborne_q":         0.80,
    "heat_q":               0.80,
    # Cardio is excluded on purpose: Phase 6 showed its val R² is ~0,
    # so any threshold would just be noise. The risk-engine has nothing
    # useful to say about cardiovascular mortality from climate alone.
}

# Map each disease to (model file, target column, threshold key, rule key).
DISEASE_TASKS = [
    ("respiratory_disease_rate", "best_respiratory.joblib", "respiratory_q", "respiratory_high"),
    ("vector_disease_risk_score", "best_vector.joblib",      "vector_q",      "vector_high"),
    ("waterborne_disease_incidents", "best_waterborne.joblib", "waterborne_q", "waterborne_high"),
    ("heat_related_admissions",   "best_heat.joblib",        "heat_q",        "heat_high"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _per_task_features(feat_cols: list[str], target: str) -> list[str]:
    """Same leak-free feature subset that step 5 used for the regressors."""
    drop = set(DISEASE_TARGETS) | {"health_impact_score", "cardio_mortality_rate"}
    return [c for c in feat_cols if c != target and c not in drop]


def _classifier_features(feat_cols: list[str]) -> list[str]:
    """Features for the overall classifier — keep climate + socio-econ, no disease leakage."""
    drop = set(DISEASE_TARGETS) | {"health_impact_score", "cardio_mortality_rate"}
    return [c for c in feat_cols if c not in drop]


def _training_quantile(y_series: pd.Series, q: float) -> float:
    """Quantile of the training target; floor at the max so an empty split is safe."""
    if len(y_series) == 0:
        return float("inf")
    return float(np.nanquantile(y_series.to_numpy(), q))


def _build_advisory(predicted_class: str, p_high: float,
                    triggered_diseases: list[str]) -> str:
    """Compose the final advisory string from the triggered rules."""
    # Header — predicted class + its probability.
    header = f"RISK: {predicted_class} (P(High)={p_high:.2f})"

    # Overall severity layer.
    severity_msg = ""
    if predicted_class == "High":
        severity_msg = ADVISORY_RULES["overall_high"]
    elif predicted_class == "Medium":
        severity_msg = ADVISORY_RULES["overall_medium"]

    # Disease-specific bodies.
    disease_msgs = [ADVISORY_RULES[k] for k in triggered_diseases]

    body_parts = [p for p in [severity_msg] + disease_msgs if p]
    if not body_parts:
        return f"{header} | Actions: routine surveillance; no disease triggers."

    return header + " | " + " ".join(body_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()
    print("[step6] loading data + saved models ...", flush=True)

    X_full, y_frame, feat_cols = _load_data()
    splits = _chronological_splits(X_full)
    train_pos = np.flatnonzero(splits["train"])
    test_pos  = np.flatnonzero(splits["test"])

    train_median = X_full.iloc[train_pos][feat_cols].median(numeric_only=True)

    # ---- overall classifier ----
    clf = joblib.load(MODEL_DIR / "best_overall_classifier.joblib")
    clf_feats = _classifier_features(feat_cols)
    # The classifier pipeline already scales internally; just feed the
    # imputed matrix.
    Xte_clf = X_full.iloc[test_pos][clf_feats].fillna(train_median)
    p_all = clf.predict_proba(Xte_clf)              # columns in clf.classes_
    classes = list(clf.classes_)                    # e.g. ["High","Low","Medium"] alphabetically
    # Argmax predicted class.
    pred_idx = p_all.argmax(axis=1)
    pred_class = np.array(classes)[pred_idx]
    high_idx = classes.index("High") if "High" in classes else pred_idx  # fallback
    p_high = p_all[:, high_idx]

    # ---- per-disease regressors ----
    triggered_counts = {k: 0 for k in ["respiratory_high", "vector_high",
                                      "waterborne_high", "heat_high"]}
    triggered_diseases_per_row: list[list[str]] = []

    for target, model_file, q_key, rule_key in DISEASE_TASKS:
        model = joblib.load(MODEL_DIR / model_file)
        task_feats = _per_task_features(feat_cols, target)
        Xte = X_full.iloc[test_pos][task_feats].fillna(train_median)
        preds = model.predict(Xte)

        # Training quantile on the actual target distribution.
        cutoff = _training_quantile(
            y_frame[target].iloc[train_pos], THRESHOLDS[q_key]
        )
        triggered_mask = preds >= cutoff
        triggered_counts[rule_key] = int(triggered_mask.sum())

        # Build / extend the per-row triggered list lazily (only on the
        # first iteration).
        if not triggered_diseases_per_row:
            triggered_diseases_per_row = [[] for _ in range(len(preds))]
        for i, hit in enumerate(triggered_mask):
            if hit:
                triggered_diseases_per_row[i].append(rule_key)

        print(f"  {target:32s} threshold={cutoff:6.2f} "
              f"(train q={THRESHOLDS[q_key]:.2f})  triggered={int(triggered_mask.sum())}/{len(preds)}",
              flush=True)

    # ---- assemble advisories ----
    n_rows = len(pred_class)
    advisories: list[str] = []
    risk_class_assigned: list[str] = []
    for i in range(n_rows):
        rd = triggered_diseases_per_row[i] if i < len(triggered_diseases_per_row) else []
        # Promote a row to "High" if its probability of High exceeds the
        # threshold OR the predicted class is already High.
        cls = str(pred_class[i])
        if cls != "High" and p_high[i] >= THRESHOLDS["class_high_prob"]:
            cls = "High"
        risk_class_assigned.append(cls)
        advisories.append(_build_advisory(cls, float(p_high[i]), rd))

    # ---- write CSV ----
    out_df = X_full.iloc[test_pos].copy()
    out_df["date"] = pd.to_datetime(out_df["date"])
    out_df["predicted_risk_class"] = risk_class_assigned
    out_df["p_high"]                = p_high
    out_df["advisory"]              = advisories
    out_df["advisory_length_chars"] = [len(a) for a in advisories]

    # Keep a tidy set of columns in the output CSV; full X stays for context.
    out_df.to_csv(OUT_CSV, index=False)

    # ---- summary ----
    summary = {
        "thresholds": THRESHOLDS,
        "n_test_rows": int(n_rows),
        "predicted_class_counts": dict(zip(*np.unique(risk_class_assigned, return_counts=True))),
        "disease_triggers": triggered_counts,
        "promoted_by_high_prob": int(
            sum(
                1 for i in range(n_rows)
                if str(pred_class[i]) != "High" and p_high[i] >= THRESHOLDS["class_high_prob"]
            )
        ),
        "any_disease_trigger": int(sum(1 for rd in triggered_diseases_per_row if rd)),
        "example_advisories": {
            "first_low":       advisories[0] if advisories else None,
            "first_high_idx":  int(np.argmax(p_high)) if n_rows else None,
            "first_high_text": advisories[int(np.argmax(p_high))] if n_rows else None,
        },
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[step6] wrote advisories  -> {OUT_CSV}", flush=True)
    print(f"[step6] wrote summary     -> {OUT_SUMMARY}", flush=True)
    print(f"[step6] runtime           {time.perf_counter()-t0:.1f}s", flush=True)
    print("\nPredicted risk-class distribution on test split:")
    for k, v in summary["predicted_class_counts"].items():
        print(f"  {k:8s} {v:5d}  ({v/n_rows*100:.1f}%)")
    print("\nDisease trigger counts (rows above training-period quantile):")
    for k, v in triggered_counts.items():
        print(f"  {k:20s} {v:5d}  ({v/n_rows*100:.1f}%)")


if __name__ == "__main__":
    main()
