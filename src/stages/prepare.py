"""Stage 1 - prepare: validate the raw CSV, apply the cleaning rules.

Out: data/interim/clean.parquet, reports/metrics/data_quality.json
"""

from __future__ import annotations

import pandas as pd

from src.core.cleaning import clean, validate_schema
from src.utils import paths
from src.utils.config import load_params
from src.utils.io import get_logger, write_json

log = get_logger("prepare")


def main() -> None:
    p = load_params()["prepare"]
    raw = pd.read_csv(paths.RAW_CSV, parse_dates=["date"])
    validate_schema(raw)
    log.info("raw: %d rows x %d columns", *raw.shape)

    df, fixed = clean(raw, p)
    log.info("negative AQI set to 0: %d", fixed["clipped_negative_to_0"]["air_quality_index"])
    log.info("healthcare access clipped to <=100: %d", fixed["clipped_to_0_100"]["healthcare_access_index"])
    log.info("duplicate %s rows dropped: %d", "/".join(p["dedup_keys"]), fixed["dropped_duplicate_keys"])

    paths.CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.CLEAN, index=False)

    write_json(paths.DATA_QUALITY, {
        "raw_rows": len(raw),
        "raw_columns": raw.shape[1],
        "raw_nulls": int(raw.isna().sum().sum()),
        "clean_rows": len(df),
        "clean_columns": df.shape[1],
        "countries": int(df["country_code"].nunique()),
        "negative_aqi_fixed": fixed["clipped_negative_to_0"]["air_quality_index"],
        "healthcare_over_100_fixed": fixed["clipped_to_0_100"]["healthcare_access_index"],
        "rows_removed": len(raw) - len(df),
        "fixed_by_rule": fixed,
    })
    log.info("wrote %s (%d rows)", paths.CLEAN.relative_to(paths.ROOT), len(df))


if __name__ == "__main__":
    main()
