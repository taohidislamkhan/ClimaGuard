"""Run the ClimaGuard pipeline end-to-end.

Goal: a single command that reproduces every metric in ``output/`` exactly.

    python run_all.py                  # full pipeline incl. SHAP
    python run_all.py --skip-slow      # stop after advisory (no SHAP)

Pipeline (each step is deterministic given the seed below):

    step1_clean_inspect.py     cleaned_data.csv
    step2_eda.py               output/charts/*.png + eda_summary.md
    step3_feature_engineering.py   featured_data.csv  (163 predictors)
    step4_feature_selection.py     feature_selection.csv + selected_features.csv
    step5_train_model.py           models/phase6/*.joblib + phase6_metrics.csv
    step6_advisory.py              output/advisories.csv
    step7_shap.py                  output/shap_importance_*.csv + shap/*.png
                                  (~30 min on this box; skipped with --skip-slow)

Determinism
-----------
Every step seeds Python, NumPy, and (where applicable) scikit-learn with
``SEED = 42``. The same input CSV therefore produces the same metrics CSV
across machines and runs.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
import runpy

import numpy as np

SEED = 42

# (module_name, optional, short label for the summary)
STEPS_FULL = [
    ("pipeline.step1_clean_inspect",     False, "clean & inspect"),
    ("pipeline.step2_eda",               False, "EDA"),
    ("pipeline.step3_feature_engineering", False, "feature engineering"),
    ("pipeline.step4_feature_selection",  False, "feature selection"),
    ("pipeline.step5_train_model",       False, "model training"),
    ("pipeline.step6_advisory",          False, "advisory rules"),
    ("pipeline.step7_shap",              True,  "SHAP explainability"),
]

STEPS_QUICK = [s for s in STEPS_FULL if not s[1]]


def _seed_everything(seed: int = SEED) -> None:
    """Pin Python, NumPy, and the relevant hash seed for full determinism.

    Note: XGBoost / scikit-learn use their own per-model ``random_state``
    argument which we set in the model factories themselves (already 42).
    This function handles the *global* state that would otherwise leak
    between steps (e.g. a stray ``np.random.choice`` in a helper).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _print_banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def _run_step(module: str, label: str) -> tuple[bool, float]:
    print(f"\n>>> [{label}] running {module}", flush=True)
    t0 = time.perf_counter()
    try:
        runpy.run_module(module, run_name="__main__")
        dt = time.perf_counter() - t0
        print(f"<<< [{label}] OK in {dt:.1f}s", flush=True)
        return True, dt
    except SystemExit as e:
        dt = time.perf_counter() - t0
        ok = (e.code is None) or (e.code == 0)
        print(f"<<< [{label}] SystemExit code={e.code} after {dt:.1f}s", flush=True)
        return ok, dt
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"<<< [{label}] FAILED after {dt:.1f}s: {type(e).__name__}: {e}",
              flush=True)
        return False, dt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-slow", action="store_true",
                        help="Stop after step 6 (skip SHAP, which takes ~30 min).")
    parser.add_argument("--from-step", type=int, default=1, metavar="N",
                        help="Resume from step N (1-indexed). Earlier steps are skipped.")
    parser.add_argument("--until-step", type=int, default=None, metavar="N",
                        help="Stop after step N (1-indexed). Later steps are skipped.")
    args = parser.parse_args(argv)

    steps = STEPS_QUICK if args.skip_slow else STEPS_FULL
    start = max(1, args.from_step)
    end = args.until_step if args.until_step is not None else len(steps)

    _print_banner("ClimaGuard pipeline orchestrator")
    print(f"Seed: {SEED}    Steps: {start}..{end} of {len(steps)}    "
          f"Skip-slow: {args.skip_slow}", flush=True)
    print(f"Python {sys.version.split()[0]}    "
          f"NumPy {np.__version__}    "
          f"Executable {sys.executable}", flush=True)

    # Seed BEFORE any step runs so first-step randomness is also pinned.
    _seed_everything(SEED)

    wall_t0 = time.perf_counter()
    log: list[dict] = []
    overall_ok = True

    for idx, (module, _optional, label) in enumerate(steps, start=1):
        if idx < start or idx > end:
            print(f"--- skipping {module} (out of range {start}..{end})", flush=True)
            continue
        # Re-seed before every step in case a previous step perturbed
        # global RNG state via an unseeded helper.
        _seed_everything(SEED)
        ok, dt = _run_step(module, label)
        log.append({"step": idx, "module": module, "label": label,
                    "ok": ok, "seconds": round(dt, 2)})
        if not ok:
            overall_ok = False
            print(f"\n!!! {module} failed — aborting.", flush=True)
            break

    total = time.perf_counter() - wall_t0
    _print_banner("Pipeline summary")
    for entry in log:
        status = "OK  " if entry["ok"] else "FAIL"
        print(f"  [{entry['step']}] {status}  "
              f"{entry['seconds']:7.2f}s  {entry['module']}  ({entry['label']})",
              flush=True)
    print(f"\nTotal wall-clock: {total:.1f}s  ({total/60:.1f} min)")
    print(f"Result: {'SUCCESS' if overall_ok else 'FAILED'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
