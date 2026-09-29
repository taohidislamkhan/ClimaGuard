"""Stage 5 - train: 4 model families x 6 tasks, winner picked on validation.

NaNs (only in the longer lags) are filled with the TRAINING median, so val and
test never leak into the imputation.

Out: models/candidates/<task>__<model>.joblib   all 24 fitted models
     models/best_<task>.joblib                  the validation winners (used by the dashboard)
     models/preprocess.json                     feature list, train medians, class labels
     reports/metrics/validation_metrics.json
"""

from __future__ import annotations

import time

import pandas as pd

from src.core import models as M
from src.utils import paths
from src.utils.config import load_params, seed_everything
from src.utils.io import get_logger, read_json, save_model, write_json

log = get_logger("train")


def load_xy(split: str, features: list[str], median: pd.Series, target: str):
    df = pd.read_parquet(paths.SPLITS[split])
    df = df[df[target].notna()]
    return df[features].fillna(median), df[target].to_numpy()


def main() -> None:
    params = load_params()
    seed, p = params["seed"], params["train"]
    seed_everything(seed)
    features = read_json(paths.SELECTED)["model_features"]

    train = pd.read_parquet(paths.SPLITS["train"])
    median = train[features].median(numeric_only=True)
    classes = sorted(train[M.CLASS_TARGET].unique().tolist())

    val_report, winners = {}, {}
    for task, kind, target in M.TASKS:
        t0 = time.perf_counter()
        Xtr, ytr = load_xy("train", features, median, target)
        Xva, yva = load_xy("val", features, median, target)
        if kind == "classification":
            family = M.classifiers(p, seed, len(classes))
            family["xgb"] = M.EncodedClassifier(family["xgb"], classes)
        else:
            family = M.regressors(p, seed)

        val_report[task] = {}
        for name, model in family.items():
            model.fit(Xtr, ytr)
            val_report[task][name] = M.score(kind, model, Xva, yva)
            save_model(model, paths.candidate(task, name))

        winners[task] = M.pick_winner(kind, val_report[task])
        save_model(family[winners[task]], paths.best_model(task))
        key = "macro_f1" if kind == "classification" else "rmse"
        log.info("%-18s winner=%-10s val %s=%.4f  (%.0fs)", task, winners[task], key,
                 val_report[task][winners[task]][key], time.perf_counter() - t0)

    write_json(paths.PREPROCESS, {"features": features, "train_median": median.to_dict(),
                                  "classes": classes, "train_end": params["split"]["train_end"]})
    write_json(paths.VALIDATION_METRICS, {
        "selection_rule": {"classification": "highest val macro_f1", "regression": "lowest val rmse"},
        "winners": winners,
        "winner_labels": {t: M.MODEL_LABELS[m] for t, m in winners.items()},
        "val": val_report,
    })


if __name__ == "__main__":
    main()
