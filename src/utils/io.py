"""Small read/write helpers shared by the stages."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib


def get_logger(stage: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    return logging.getLogger(stage)


def _clean(obj):
    """Make NaN/inf JSON-safe (null) and numpy scalars plain Python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF on every OS, so Git-tracked outputs hash the same after a clone.
    path.write_text(json.dumps(_clean(obj), indent=indent, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: Path):
    """Load a model and predict single-threaded.

    Random Forests trained with n_jobs=-1 also predict in parallel, summing the
    trees in thread order, so the last float digit can change between runs.
    One thread keeps every metric bit-for-bit reproducible.
    """
    model = joblib.load(path)
    for step in getattr(model, "steps", [(None, model)]):
        est = getattr(step[1], "model", step[1])
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
    return model
