"""Step 1 — Clean and inspect the raw climate-health dataset.

Cleaning actions (Phase 2):
  1. Clamp `air_quality_index` and `pm25_ugm3` to >= 0.
  2. Clip all disease rate / count columns to >= 0.
  3. Force `drought_indicator` and `flood_indicator` to strict 0/1.
  4. Clip `healthcare_access_index` to [0, 100].
  5. Sort by country, then date (required before any lag/rolling step).
  6. Derive `season` (hemisphere-aware) and `climate_zone` (latitude-based).

Compliance note:
  Humidity, elevation, and named case counts are absent from the raw file. We
  deliberately do NOT fabricate them — the project guidelines require that
  every feature be publicly available and documented. Anything missing stays
  missing; this limitation is recorded in `data_card.md`.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/global_climate_health_impact_tracker_2015_2025.csv")
OUT_PATH = Path("data/processed/cleaned_data.csv")

# Columns that must be non-negative (rates, counts, concentrations).
NON_NEGATIVE_COLS = [
    "air_quality_index",
    "pm25_ugm3",
    "respiratory_disease_rate",
    "cardio_mortality_rate",
    "vector_disease_risk_score",
    "waterborne_disease_incidents",
    "heat_related_admissions",
    "heat_wave_days",
    "extreme_weather_events",
    "precipitation_mm",
]

# Columns that must be strictly {0, 1}.
BINARY_INDICATOR_COLS = [
    "drought_indicator",
    "flood_indicator",
]

# Columns bounded to [0, 100] (index scores).
INDEX_0_100_COLS = [
    "healthcare_access_index",
    "mental_health_index",
    "food_security_index",
]


def inspect(df: pd.DataFrame) -> None:
    """Phase-1 inspection summary."""
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nNulls per column:\n", df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print(
        "\nDuplicate country-week keys:",
        int(df.duplicated(subset=["country_code", "year", "week"]).sum()),
    )
    print(
        "\nAQI < 0:", int((df["air_quality_index"] < 0).sum()),
        "| PM2.5 < 0:", int((df["pm25_ugm3"] < 0).sum()),
        "| healthcare_access > 100:", int((df["healthcare_access_index"] > 100).sum()),
        "| drought_indicator not in {0,1}:",
        int(~df["drought_indicator"].isin([0, 1]).sum()),
        "| flood_indicator not in {0,1}:",
        int(~df["flood_indicator"].isin([0, 1]).sum()),
    )


def hemisphere(lat: float) -> str:
    return "south" if lat < 0 else "north"


def derive_season(month: int, lat: float) -> str:
    """Hemisphere-aware meteorological season (Dec-Feb is summer south of equator)."""
    if pd.isna(month):
        return "unknown"
    south = lat < 0
    if month in (12, 1, 2):
        northern = "winter"
    elif month in (3, 4, 5):
        northern = "spring"
    elif month in (6, 7, 8):
        northern = "summer"
    else:
        northern = "autumn"
    if south:
        flip = {"winter": "summer", "spring": "autumn", "summer": "winter", "autumn": "spring"}
        return flip[northern]
    return northern


def derive_climate_zone(lat: float) -> str:
    """Latitude-based climate zone."""
    a = abs(lat)
    if a <= 23.5:
        return "tropical"
    if a <= 35:
        return "subtropical"
    if a <= 60:
        return "temperate"
    return "subpolar"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Clamp non-negative physical / count columns.
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            n = int((df[col] < 0).sum())
            df[col] = df[col].clip(lower=0)
            if n:
                print(f"  Clipped {n} negative '{col}' rows to 0.")

    # 2. Force binary indicator columns to strict 0/1.
    for col in BINARY_INDICATOR_COLS:
        if col in df.columns:
            n = int((~df[col].isin([0, 1])).sum())
            df[col] = df[col].clip(lower=0, upper=1).round().astype("int64")
            if n:
                print(f"  Forced {n} '{col}' values into {{0,1}}.")

    # 3. Clip bounded indices to [0, 100].
    for col in INDEX_0_100_COLS:
        if col in df.columns:
            n = int(((df[col] < 0) | (df[col] > 100)).sum())
            df[col] = df[col].clip(lower=0, upper=100)
            if n:
                print(f"  Clipped {n} '{col}' rows into [0, 100].")

    # 4. Drop duplicates.
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {before - len(df)} full-row duplicates.")

    before = len(df)
    df = df.drop_duplicates(subset=["country_code", "year", "week"], keep="first")
    print(f"  Dropped {before - len(df)} duplicate country-week rows.")

    # 5. Sort by country, then date (mandatory before any lag/rolling window).
    df = df.sort_values(["country_code", "date"]).reset_index(drop=True)

    # 6. Derive helper columns.
    df["hemisphere"] = df["latitude"].apply(hemisphere)
    df["season"] = df.apply(lambda r: derive_season(r["month"], r["latitude"]), axis=1)
    df["climate_zone"] = df["latitude"].apply(derive_climate_zone)

    return df


def main() -> None:
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    print("=== Phase 1: Inspection ===")
    inspect(df)

    print("\n=== Phase 2: Cleaning ===")
    df = clean(df)
    print("\nCleaned shape:", df.shape)
    print("Added columns: hemisphere, season, climate_zone")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
