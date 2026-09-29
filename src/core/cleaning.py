"""Cleaning rules for the raw climate-health file (was pipeline/step1).

Humidity, elevation and named case counts are absent from the raw file. We do
not fabricate them; the gap is documented in the README and data_card.md.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "record_id": "number", "country_code": "text", "country_name": "text", "region": "text",
    "income_level": "text", "date": "text", "year": "number", "month": "number", "week": "number",
    "latitude": "number", "longitude": "number", "population_millions": "number",
    "temperature_celsius": "number", "temp_anomaly_celsius": "number", "precipitation_mm": "number",
    "heat_wave_days": "number", "drought_indicator": "number", "flood_indicator": "number",
    "extreme_weather_events": "number", "pm25_ugm3": "number", "air_quality_index": "number",
    "respiratory_disease_rate": "number", "cardio_mortality_rate": "number",
    "vector_disease_risk_score": "number", "waterborne_disease_incidents": "number",
    "heat_related_admissions": "number", "healthcare_access_index": "number",
    "gdp_per_capita_usd": "number", "mental_health_index": "number", "food_security_index": "number",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if a column is missing or has the wrong kind of values."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")
    wrong = [c for c, kind in REQUIRED_COLUMNS.items()
             if kind == "number" and not pd.api.types.is_numeric_dtype(df[c])]
    if wrong:
        raise ValueError(f"Columns expected to be numeric are not: {wrong}")


def hemisphere(lat: float) -> str:
    return "south" if lat < 0 else "north"


def derive_season(month: int, lat: float) -> str:
    """Hemisphere-aware meteorological season (Dec-Feb is summer south of the equator)."""
    if pd.isna(month):
        return "unknown"
    if month in (12, 1, 2):
        northern = "winter"
    elif month in (3, 4, 5):
        northern = "spring"
    elif month in (6, 7, 8):
        northern = "summer"
    else:
        northern = "autumn"
    if lat < 0:
        return {"winter": "summer", "spring": "autumn", "summer": "winter", "autumn": "spring"}[northern]
    return northern


def derive_climate_zone(lat: float) -> str:
    a = abs(lat)
    if a <= 23.5:
        return "tropical"
    if a <= 35:
        return "subtropical"
    if a <= 60:
        return "temperate"
    return "subpolar"


def clean(df: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules. Returns the cleaned frame and the count fixed by each rule."""
    df = df.copy()
    fixed: dict[str, dict[str, int]] = {"clipped_negative_to_0": {}, "forced_binary": {},
                                        "clipped_to_0_100": {}}

    for col in p["non_negative_cols"]:
        fixed["clipped_negative_to_0"][col] = int((df[col] < 0).sum())
        df[col] = df[col].clip(lower=0)

    for col in p["binary_cols"]:
        fixed["forced_binary"][col] = int((~df[col].isin([0, 1])).sum())
        df[col] = df[col].clip(lower=0, upper=1).round().astype("int64")

    for col in p["index_0_100_cols"]:
        fixed["clipped_to_0_100"][col] = int(((df[col] < 0) | (df[col] > 100)).sum())
        df[col] = df[col].clip(lower=0, upper=100)

    before = len(df)
    df = df.drop_duplicates()
    fixed["dropped_full_duplicates"] = before - len(df)

    # ISO week numbering repeats a week at some year boundaries; keep the first row.
    before = len(df)
    df = df.drop_duplicates(subset=p["dedup_keys"], keep="first")
    fixed["dropped_duplicate_keys"] = before - len(df)

    # Sort by country, then date: required before any lag / rolling window.
    df = df.sort_values(["country_code", "date"]).reset_index(drop=True)

    df["hemisphere"] = df["latitude"].apply(hemisphere)
    df["season"] = df.apply(lambda r: derive_season(r["month"], r["latitude"]), axis=1)
    df["climate_zone"] = df["latitude"].apply(derive_climate_zone)
    return df, fixed
