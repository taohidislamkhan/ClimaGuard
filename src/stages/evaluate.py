"""Stage 6 - evaluate: score all 24 models on the held-out test split.

Out (metrics, tracked by Git):
    reports/metrics/classifier_metrics.json   accuracy / macro-F1 / ROC-AUC per model + winner
    reports/metrics/regressor_metrics.json    R² / MAE / RMSE per target and model + winner
    reports/metrics/model_comparison.csv      every model x {val, test} (the dashboard table)
Out (plots):
    reports/plots/confusion_matrix.csv        actual vs predicted class (winner)
    reports/plots/pred_vs_actual/<task>.csv   actual vs predicted value (winner)
    reports/plots/*.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.core import models as M
from src.utils import paths
from src.utils.io import get_logger, load_model, read_json, write_json

log = get_logger("evaluate")
CSV_METRIC_NAMES = {"roc_auc": "roc_auc_ovr_weighted"}   # column names the dashboard expects


def main() -> None:
    pre = read_json(paths.PREPROCESS)
    val = read_json(paths.VALIDATION_METRICS)
    features, median = pre["features"], pd.Series(pre["train_median"])
    test = pd.read_parquet(paths.SPLITS["test"])
    X = test[features].fillna(median)
    n_rows = {"val": len(pd.read_parquet(paths.SPLITS["val"])), "test": len(test)}

    clf_out, reg_out, rows = {}, {}, []
    pva_dir = paths.PLOTS / "pred_vs_actual"
    pva_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for (task, kind, target), ax in zip(M.TASKS, axes.flat):
        y = test[target].to_numpy()
        winner = val["winners"][task]
        per_model = {}
        for name in val["val"][task]:
            per_model[name] = M.score(kind, load_model(paths.candidate(task, name)), X, y)
            for split, m in (("val", val["val"][task][name]), ("test", per_model[name])):
                rows.append({"task": task, "target": target, "model": name, "split": split,
                             "n_rows": n_rows[split],
                             **{CSV_METRIC_NAMES.get(k, k): v for k, v in m.items()}})

        best = load_model(paths.best_model(task))
        pred = best.predict(X)
        if kind == "classification":
            clf_out = {"winner": per_model[winner], **per_model}
            pd.DataFrame({"actual": y, "predicted": pred}).to_csv(paths.PLOTS / "confusion_matrix.csv", index=False)
            ConfusionMatrixDisplay.from_predictions(y, pred, labels=["Low", "Medium", "High"], ax=ax,
                                                    colorbar=False, cmap="Blues")
            ax.set_title(f"Overall risk class ({M.MODEL_LABELS[winner]})")
            log.info("%-18s %s: acc=%.3f f1=%.3f auc=%.3f", task, winner,
                     *(per_model[winner][k] for k in ("accuracy", "macro_f1", "roc_auc")))
        else:
            reg_out[task] = {"winner": per_model[winner], **per_model}
            pd.DataFrame({"actual": y, "predicted": np.round(pred, 4)}).to_csv(pva_dir / f"{task}.csv", index=False)
            ax.scatter(y, pred, s=4, alpha=0.35)
            lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
            ax.plot([lo, hi], [lo, hi], color="grey", lw=1)
            ax.set(title=f"{task} ({M.MODEL_LABELS[winner]}, R² {per_model[winner]['r2']:.3f})",
                   xlabel="actual", ylabel="predicted")
            log.info("%-18s %s: r2=%.3f rmse=%.3f", task, winner, per_model[winner]["r2"],
                     per_model[winner]["rmse"])

    fig.tight_layout()
    fig.savefig(paths.PLOTS / "test_performance.png", dpi=110)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(paths.MODEL_COMPARISON, index=False)
    write_json(paths.CLASSIFIER_METRICS, clf_out)
    write_json(paths.REGRESSOR_METRICS, reg_out)


if __name__ == "__main__":
    main()
