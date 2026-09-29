"""Regenerate the README's result tables from the metrics JSON.

    python -m src.utils.readme_tables          # rewrite the tables in README.md
    python -m src.utils.readme_tables --check  # exit 1 if README.md is out of date

The tables sit between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->``.
"""

from __future__ import annotations

import sys

from src.core.models import MODEL_LABELS
from src.utils import paths
from src.utils.io import read_json

README = paths.ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
TASK_LABELS = {"respiratory": "Respiratory", "vector": "Vector-borne", "heat": "Heat-related",
               "waterborne": "Waterborne", "cardio": "Cardiovascular"}


def render() -> str:
    winners = read_json(paths.VALIDATION_METRICS)["winners"]
    clf = read_json(paths.CLASSIFIER_METRICS)
    reg = read_json(paths.REGRESSOR_METRICS)
    drift = read_json(paths.DRIFT_REPORT)
    shap = read_json(paths.SHAP / "summary.json")

    out = ["**Overall risk classifier** (Low / Medium / High), test split 2024-01-07 → 2025-10-19:", "",
           "| Model | Accuracy | Macro-F1 | ROC-AUC (OvR, weighted) |", "|---|---|---|---|"]
    for name in ("logreg", "rf", "xgb", "dt"):
        m = clf[name]
        star = " ★ selected" if name == winners["overall_classifier"] else ""
        out.append(f"| {MODEL_LABELS[name]}{star} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['roc_auc']:.3f} |")

    out += ["", "**Disease regressors** (validation winner per disease), test split:", "",
            "| Disease | Model | R² | MAE | RMSE | Random Forest R² |", "|---|---|---|---|---|---|"]
    for task, label in TASK_LABELS.items():
        w, rf = reg[task]["winner"], reg[task]["rf"]
        out.append(f"| {label} | {MODEL_LABELS[winners[task]]} | {w['r2']:.3f} | {w['mae']:.2f} | "
                   f"{w['rmse']:.2f} | {rf['r2']:.3f} |")

    out += ["", "**Top SHAP features** (mean |SHAP| of the saved winner on test weeks):", ""]
    for task, s in shap.items():
        out.append(f"- **{task}** ({s['model']}): " + ", ".join(f"`{f}`" for f in s["top5"][:3]))

    d, c = drift["data_drift"], drift["concept_drift"]
    out += ["", "**Drift** (train 2015–2022 vs test 2024–2025):", "",
            f"- Data drift: {d['n_drifted']} of {d['n_features']} features drifted; "
            f"{d['top_features_drifted']} of the top {d['top_features_checked']} SHAP features. "
            f"Largest PSI: `{d['worst'][0]['feature']}` = {d['worst'][0]['psi']:.3f}.",
            f"- Concept drift: {c['n_flagged']} of {c['n_windows']} task-quarters worse than the same quarter "
            f"of 2023 beyond tolerance.",
            f"- `retrain_recommended`: **{str(drift['retrain_recommended']).lower()}**"]
    return "\n".join(out)


def main() -> None:
    text = README.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render()}\n{END}{tail}"
    if "--check" in sys.argv:
        if new != text:
            sys.exit("README.md results are out of date: run `python -m src.utils.readme_tables`.")
        print("README.md results are up to date.")
        return
    README.write_text(new, encoding="utf-8", newline="\n")
    print("README.md results updated.")


if __name__ == "__main__":
    main()
