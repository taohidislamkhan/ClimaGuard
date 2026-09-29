"""Stage 3 - select_features: correlation pruning + committee vote of three rankings.

Out: data/processed/selected_features.json, data/processed/feature_rankings.csv,
     reports/metrics/feature_selection.json
"""

from __future__ import annotations

import pandas as pd

from src.core.features import drop_first_week
from src.core.models import model_features
from src.core.selection import select
from src.utils import paths
from src.utils.config import load_params, seed_everything
from src.utils.io import get_logger, write_json

log = get_logger("select_features")


def main() -> None:
    params = load_params()
    seed_everything(params["seed"])
    df = drop_first_week(pd.read_parquet(paths.FEATURES))

    committee, table, summary = select(df, params["select"], params["seed"])
    features = model_features(committee)

    table.to_csv(paths.RANKINGS, index=False)
    write_json(paths.SELECTED, {"committee": committee, "model_features": features})
    write_json(paths.FEATURE_SELECTION, {**summary, "model_features": len(features)})
    log.info("%d candidates -> %d after pruning -> %d in committee -> %d model features",
             summary["candidate_features"], summary["after_dedup"], len(committee), len(features))


if __name__ == "__main__":
    main()
