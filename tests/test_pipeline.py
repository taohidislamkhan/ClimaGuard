"""Checks on the DVC pipeline outputs and the pure helpers behind them."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.stages.drift import psi
from src.utils import paths
from src.utils.config import load_params

needs_outputs = pytest.mark.skipif(not paths.SPLIT_SUMMARY.exists(), reason="run `dvc repro` first")


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, ref.copy(), 10) < 1e-6
    assert psi(ref, ref + 1.0, 10) > 0.2


def test_psi_handles_binary_features():
    ref = np.array([0] * 900 + [1] * 100, dtype=float)
    assert psi(ref, ref.copy(), 10) < 1e-6
    assert psi(ref, np.array([0] * 500 + [1] * 500, dtype=float), 10) > 0.2


@needs_outputs
def test_data_quality_counts():
    dq = json.loads(paths.DATA_QUALITY.read_text())
    assert dq["raw_rows"] == 14100 and dq["raw_columns"] == 30
    assert dq["negative_aqi_fixed"] == 374
    assert dq["clean_rows"] == 14050


@needs_outputs
def test_split_is_chronological_and_disjoint():
    p = load_params()["split"]
    parts = {s: pd.read_parquet(f, columns=["date", "record_id"]) for s, f in paths.SPLITS.items()}
    assert parts["train"]["date"].max() <= pd.Timestamp(p["train_end"]) < parts["val"]["date"].min()
    assert parts["val"]["date"].max() <= pd.Timestamp(p["val_end"]) < parts["test"]["date"].min()
    ids = [set(df["record_id"]) for df in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


@needs_outputs
def test_readme_results_match_metrics():
    from src.utils.readme_tables import END, README, START, render
    text = README.read_text(encoding="utf-8")
    assert text.split(START, 1)[1].split(END, 1)[0].strip() == render().strip(), \
        "run `python -m src.utils.readme_tables`"


@needs_outputs
def test_no_target_is_a_model_feature():
    features = json.loads(paths.SELECTED.read_text())["model_features"]
    from src.core.models import ALL_TARGETS
    assert not set(features) & set(ALL_TARGETS)
