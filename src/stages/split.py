"""Stage 4 - split: chronological train / validation / test.

train <= split.train_end < val <= split.val_end < test. No shuffling: the
rolling and lag features would otherwise leak future weeks into training.

Out: data/processed/{train,val,test}.parquet, reports/metrics/split_summary.json
"""

from __future__ import annotations

import pandas as pd

from src.core.features import drop_first_week
from src.core.models import ALL_TARGETS
from src.utils import paths
from src.utils.config import load_params
from src.utils.io import get_logger, read_json, write_json

log = get_logger("split")
KEYS = ["record_id", "date", "year", "country_name"]


def main() -> None:
    p = load_params()["split"]
    features = read_json(paths.SELECTED)["model_features"]
    df = drop_first_week(pd.read_parquet(paths.FEATURES))
    df = df[KEYS + features + ALL_TARGETS]

    train_end, val_end = pd.Timestamp(p["train_end"]), pd.Timestamp(p["val_end"])
    masks = {
        "train": df["date"] <= train_end,
        "val": (df["date"] > train_end) & (df["date"] <= val_end),
        "test": df["date"] > val_end,
    }
    summary = {}
    for name, mask in masks.items():
        part = df[mask].reset_index(drop=True)
        if part.empty:
            raise ValueError(f"The {name} split is empty; check split.train_end / split.val_end.")
        part.to_parquet(paths.SPLITS[name], index=False)
        summary[name] = {"rows": len(part), "start": str(part["date"].min().date()),
                         "end": str(part["date"].max().date())}
        log.info("%-5s %5d rows  %s -> %s", name, len(part), summary[name]["start"], summary[name]["end"])
    write_json(paths.SPLIT_SUMMARY, summary)


if __name__ == "__main__":
    main()
