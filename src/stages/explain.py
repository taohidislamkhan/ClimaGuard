"""Stage 7 - explain: global SHAP for each saved winner on a test sample.

Linear winners (scaled Logistic Regression / ElasticNet) get exact linear SHAP
(coef x standardised x). Tree winners use shap.TreeExplainer. For the
classifier, values are for the High class.

Out: reports/shap/<task>.json   mean |SHAP| of every feature
     reports/shap/<task>.png    bar chart + beeswarm
     reports/shap/summary.json  top features per task
"""

from __future__ import annotations

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.core import models as M
from src.utils import paths
from src.utils.config import load_params
from src.utils.io import get_logger, load_model, read_json, write_json

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
log = get_logger("explain")


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """(rows, features) SHAP matrix; High class for classifiers."""
    steps = getattr(model, "steps", None)
    final = steps[-1][1] if steps else getattr(model, "model", model)
    classes = [str(c) for c in getattr(model, "classes_", [])]
    high = classes.index("High") if "High" in classes else None
    if steps and hasattr(final, "coef_"):
        z = np.asarray(model[:-1].transform(X), dtype=float)
        coef = np.asarray(final.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[high] if coef.shape[0] > 1 else coef[0]
        return z * coef
    sv = shap.TreeExplainer(final).shap_values(X)
    if isinstance(sv, list):
        return np.asarray(sv[high])
    sv = np.asarray(sv)
    return sv[:, :, high] if sv.ndim == 3 else sv


def plot(task: str, label: str, sv: np.ndarray, X: pd.DataFrame, top_k: int) -> None:
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(2, 1, 1)
    ax.barh(range(len(order))[::-1], mean_abs[order], color="#4575b4")
    ax.set_yticks(range(len(order))[::-1], X.columns[order], fontsize=9)
    ax.set(xlabel="mean |SHAP|", title=f"{task} ({label}): global importance")
    fig.add_subplot(2, 1, 2)
    shap.summary_plot(sv, X, max_display=top_k, show=False, plot_size=None)
    plt.title(f"{task}: direction and size of each feature's effect")
    fig.tight_layout()
    fig.savefig(paths.SHAP / f"{task}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    params = load_params()
    p, seed = params["explain"], params["seed"]
    pre = read_json(paths.PREPROCESS)
    winners = read_json(paths.VALIDATION_METRICS)["winners"]
    test = pd.read_parquet(paths.SPLITS["test"])
    X = test[pre["features"]].fillna(pd.Series(pre["train_median"]))
    sample = X.sample(n=min(p["sample_size"], len(X)), random_state=seed)

    paths.SHAP.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task, kind, target in M.TASKS:
        t0 = time.perf_counter()
        sv = shap_values(load_model(paths.best_model(task)), sample)
        imp = pd.DataFrame({"feature": sample.columns, "mean_abs": np.abs(sv).mean(axis=0),
                            "std_abs": np.abs(sv).std(axis=0)})
        imp = imp.sort_values("mean_abs", ascending=False).reset_index(drop=True)
        label = M.MODEL_LABELS[winners[task]]
        write_json(paths.SHAP / f"{task}.json", {
            "task": task, "target": target, "model": winners[task], "model_label": label,
            "n_rows": len(sample),
            "unit": "High-class log-odds" if kind == "classification" and winners[task] == "logreg"
                    else "High-class probability" if kind == "classification" else f"{target} units",
            "importance": imp.round(6).to_dict("records"),
        })
        plot(task, label, sv, sample, p["top_k"])
        summary[task] = {"model": label, "top5": imp["feature"].head(5).tolist()}
        log.info("%-18s %s: top = %s  (%.0fs)", task, label, ", ".join(summary[task]["top5"][:3]),
                 time.perf_counter() - t0)
    write_json(paths.SHAP / "summary.json", summary)


if __name__ == "__main__":
    main()
