"""Stage 2 - featurize: country-grouped rolling windows, lags, cyclical
encodings, interactions, dummies, and the Low/Medium/High label.

Out: data/processed/features.parquet, reports/metrics/featurize.json
"""

from __future__ import annotations

import pandas as pd

from src.core.features import build_features
from src.utils import paths
from src.utils.config import load_params
from src.utils.io import get_logger, write_json

log = get_logger("featurize")


def main() -> None:
    params = load_params()
    df = pd.read_parquet(paths.CLEAN)
    df, summary = build_features(df, params["featurize"], params["label"])

    df.to_parquet(paths.FEATURES, index=False)
    write_json(paths.FEATURIZE_SUMMARY, summary)
    log.info("features: %d rows, %d predictors", summary["rows"], summary["n_predictors"])
    log.info("risk-class cut points (years <= %d): %s",
             params["label"]["train_end_year"], [round(e, 3) for e in summary["risk_class_edges"]])


if __name__ == "__main__":
    main()
