"""Stage 8 - drift: is the test period (2024-25) still like the training period?

Data drift     each model feature, train (reference) vs test (current):
               PSI on quantile bins of the reference + two-sample KS test.
               A feature has drifted when PSI >= drift.psi_threshold AND the
               KS p-value < drift.ks_pvalue.
Concept drift  the winners' error per test quarter, compared with the SAME
               calendar quarter of the validation year (2024Q3 vs 2023Q3), so
               seasonality does not look like drift. The classifier is flagged
               when macro-F1 drops by more than drift.f1_tolerance; a regressor
               when RMSE rises by more than drift.rmse_tolerance (relative).
               RMSE rather than R²: within one quarter the target varies
               little (heat admissions are ~0 all winter), so R² is unstable.
Response       retrain_recommended = too many top features drifted, or the
               share of flagged quarters >= drift.max_flagged_share. The policy: move split.train_end / val_end
               forward in params.yaml and run `dvc repro`.

Out: reports/drift/drift_report.json   (metrics)
     reports/drift/feature_drift.csv
     reports/plots/drift_quarterly.csv (plots)
     reports/drift/drift.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.core import models as M
from src.utils import paths
from src.utils.config import load_params
from src.utils.io import get_logger, load_model, read_json, write_json

log = get_logger("drift")
EPS = 1e-4


def psi(ref: np.ndarray, cur: np.ndarray, bins: int) -> float:
    """Population Stability Index with bins taken from the reference period."""
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    values = np.unique(ref)
    if len(values) <= bins:              # binary / few levels: one bin per level
        edges = np.concatenate([[-np.inf], (values[:-1] + values[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + EPS
    c = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def data_drift(train: pd.DataFrame, test: pd.DataFrame, features: list[str], p: dict) -> pd.DataFrame:
    rows = []
    for f in features:
        ref, cur = train[f].to_numpy(float), test[f].to_numpy(float)
        ks = ks_2samp(ref[~np.isnan(ref)], cur[~np.isnan(cur)])
        rows.append({"feature": f, "psi": psi(ref, cur, p["psi_bins"]), "ks_stat": float(ks.statistic),
                     "ks_pvalue": float(ks.pvalue), "train_mean": float(np.nanmean(ref)),
                     "test_mean": float(np.nanmean(cur))})
    df = pd.DataFrame(rows)
    df["drifted"] = (df["psi"] >= p["psi_threshold"]) & (df["ks_pvalue"] < p["ks_pvalue"])
    return df.sort_values("psi", ascending=False).reset_index(drop=True)


def quarterly_scores(model, kind: str, df: pd.DataFrame, X: pd.DataFrame, target: str) -> dict:
    """{quarter label: metric} for one split."""
    quarter = df["date"].dt.to_period("Q")
    out = {}
    for q in sorted(quarter.unique()):
        m = (quarter == q).to_numpy()
        y, pred = df.loc[m, target].to_numpy(), model.predict(X[m])
        if kind == "classification":
            out[q] = M.classifier_metrics(y, pred, model.predict_proba(X[m]))["macro_f1"]
        else:
            out[q] = M.regressor_metrics(y, pred)["rmse"]
    return out


def concept_drift(val_df, X_val, test, X_test, p: dict) -> pd.DataFrame:
    rows = []
    for task, kind, target in M.TASKS:
        if task not in p["concept_tasks"]:
            continue
        model = load_model(paths.best_model(task))
        base = {q.quarter: v for q, v in quarterly_scores(model, kind, val_df, X_val, target).items()}
        for q, score in quarterly_scores(model, kind, test, X_test, target).items():
            ref = base[q.quarter]
            if kind == "classification":
                change, flagged = score - ref, score < ref - p["f1_tolerance"]
            else:
                change, flagged = score / ref - 1, score > ref * (1 + p["rmse_tolerance"])
            rows.append({"task": task, "quarter": str(q), "metric": "macro_f1" if kind == "classification"
                         else "rmse", "score": score, "val_same_quarter": ref, "change": change,
                         "flagged": bool(flagged)})
    return pd.DataFrame(rows)


def main() -> None:
    p = load_params()["drift"]
    pre = read_json(paths.PREPROCESS)
    features = pre["features"]
    median = pd.Series(pre["train_median"])
    train = pd.read_parquet(paths.SPLITS["train"])
    val_df = pd.read_parquet(paths.SPLITS["val"])
    test = pd.read_parquet(paths.SPLITS["test"])

    feat = data_drift(train, test, features, p)
    shap_rank = [r["feature"] for r in read_json(paths.SHAP / "overall_classifier.json")["importance"]]
    top = shap_rank[: p["top_n_features"]]
    top_drifted = feat[feat["feature"].isin(top) & feat["drifted"]]["feature"].tolist()
    share = len(top_drifted) / len(top)

    perf = concept_drift(val_df, val_df[features].fillna(median), test, test[features].fillna(median), p)
    flagged = perf[perf["flagged"]]
    flagged_share = len(flagged) / len(perf)

    reasons = []
    if share >= p["max_drifted_share"]:
        reasons.append(f"{len(top_drifted)} of the top {len(top)} model features drifted "
                       f"(share {share:.2f} >= {p['max_drifted_share']})")
    if flagged_share >= p["max_flagged_share"]:
        reasons.append(f"{len(flagged)} of {len(perf)} task-quarters are worse than the same quarter of "
                       f"the validation year (share {flagged_share:.2f} >= {p['max_flagged_share']})")

    paths.DRIFT.mkdir(parents=True, exist_ok=True)
    feat.round(6).to_csv(paths.DRIFT / "feature_drift.csv", index=False, lineterminator="\n")
    perf.to_csv(paths.PLOTS / "drift_quarterly.csv", index=False, lineterminator="\n")
    write_json(paths.DRIFT_REPORT, {
        "reference": "train split", "current": "test split",
        "data_drift": {
            "n_features": len(feat), "n_drifted": int(feat["drifted"].sum()),
            "top_features_checked": len(top), "top_features_drifted": len(top_drifted),
            "top_drifted_share": round(share, 4), "top_drifted": top_drifted,
            "max_psi": round(float(feat["psi"].max()), 4),
            "worst": feat.head(5)[["feature", "psi", "ks_pvalue"]].round(4).to_dict("records"),
        },
        "concept_drift": {
            "n_windows": len(perf), "n_flagged": len(flagged), "flagged_share": round(flagged_share, 4),
            "flagged_by_task": flagged.groupby("task").size().to_dict(),
            "flagged": flagged[["task", "quarter", "metric", "score", "val_same_quarter", "change"]]
                       .round(4).to_dict("records"),
        },
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
        "response": "Move split.train_end and split.val_end forward in params.yaml so the newest "
                    "weeks are learned from, then run `dvc repro` and compare with `dvc metrics diff`.",
        "thresholds": p,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top_feat = feat.head(15)
    axes[0].barh(top_feat["feature"][::-1], top_feat["psi"][::-1],
                 color=np.where(top_feat["drifted"][::-1], "#E5484D", "#4575b4"))
    axes[0].axvline(p["psi_threshold"], color="grey", ls="--")
    axes[0].set(title="Data drift: PSI, train vs test (red = drifted)", xlabel="PSI")
    for task, g in perf.groupby("task"):
        worse = -g["change"] if task == "overall_classifier" else g["change"]
        axes[1].plot(g["quarter"], worse, marker="o", label=task)
    axes[1].axhline(p["rmse_tolerance"], color="grey", ls="--", label="RMSE tolerance")
    axes[1].axhline(p["f1_tolerance"], color="grey", ls=":", label="F1 tolerance")
    axes[1].set(title="Concept drift: how much worse than the same quarter of 2023",
                ylabel="RMSE increase (relative) / F1 drop (absolute)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.DRIFT / "drift.png", dpi=110)
    plt.close(fig)

    log.info("data drift: %d/%d features; top-%d drifted: %s", feat["drifted"].sum(), len(feat), len(top),
             top_drifted or "none")
    log.info("concept drift: %d/%d quarters flagged", len(flagged), len(perf))
    log.info("retrain_recommended = %s", bool(reasons))


if __name__ == "__main__":
    main()
