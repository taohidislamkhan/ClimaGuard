# ClimaGuard — Data Card (Phase 1, Day 1–2)

**Source:** `data/raw/global_climate_health_impact_tracker_2015_2025.csv`
**Inspected:** 2026-09-18
**Compliance status:** Documented per project guidelines (dataset card requirement).

## 1. Shape & identity

| Property | Value |
|---|---|
| Rows | **14,100** |
| Columns | **30** |
| Countries | **25** (e.g. United States, India, China, Brazil, Nigeria …) |
| Time unit | Weekly (564 weeks × 25 countries = 14,100 country-week rows) |
| Date range | **2015-01-04 → 2025-10-19** |
| Duplicates (full row) | **0** |
| Duplicate country-week keys | **0** |

## 2. Schema

```
record_id                       int64
country_code                    object (ISO-3)
country_name                    object
region                          object
income_level                    object (Low / Lower-Middle / Upper-Middle / High)
date                            object (YYYY-MM-DD)
year                            int64
month                           int64
week                            int64
latitude                        float64
longitude                       float64
population_millions             float64
temperature_celsius             float64
temp_anomaly_celsius            float64
precipitation_mm                float64
heat_wave_days                  float64
drought_indicator               float64
flood_indicator                 float64
extreme_weather_events          float64
pm25_ugm3                       float64
air_quality_index               float64
respiratory_disease_rate        float64
cardio_mortality_rate           float64
vector_disease_risk_score       float64
waterborne_disease_incidents    float64
heat_related_admissions         float64
healthcare_access_index         float64  (expected 0–100)
gdp_per_capita_usd              float64
mental_health_index             float64  (expected 0–100)
food_security_index             float64  (expected 0–100)
```

## 3. Missingness

`df.isnull().sum()` returns **0 for every column**. No imputation required for nulls.

## 4. Physical / domain-validity checks

| Check | Expected | Observed | Rows affected |
|---|---|---|---|
| `air_quality_index < 0` | 0 | **374** | 374 rows |
| `air_quality_index > 500` | 0 | 0 | — |
| `healthcare_access_index > 100` | 0 | **3** | 3 rows |
| `healthcare_access_index < 0` | 0 | 0 | — |
| `temperature_celsius` range | plausible | −20.74 → 38.33 °C | — |
| `pm25_ugm3 < 0` | 0 | 0 | — |
| `mental_health_index < 0` | 0 | 0 | — |
| `food_security_index > 100` | 0 | 0 | — |

### Flagged data-quality issues (must be cleaned in Step 1)

1. **374 negative AQI readings** — AQI is physically non-negative. Treat as sensor error / sign-flip; will be clipped to 0 (or removed) in the `prepare` stage (`src/core/cleaning.py`).
2. **3 `healthcare_access_index` values > 100** — index is bounded [0, 100]. Will be clipped to 100.

## 5. Inspection commands (reproducible)

```python
import pandas as pd
df = pd.read_csv("data/raw/global_climate_health_impact_tracker_2015_2025.csv")
print(df.shape, df.dtypes, df.isnull().sum(), df.duplicated().sum())
print((df['air_quality_index'] < 0).sum())                # 374
print((df['healthcare_access_index'] > 100).sum())        # 3
print(df['date'].min(), df['date'].max())                 # 2015-01-04 2025-10-19
print(df['country_name'].nunique())                       # 25
```

## 6. Compliance summary

| Guideline | Status |
|---|---|
| Dataset documented (`data_card.md`) | ✅ |
| Shape / dtypes / nulls / duplicates inspected | ✅ |
| Impossible values flagged | ✅ |
| Cleaning actions queued for next step | ✅ (clip AQI ≥ 0, clip healthcare_access ≤ 100) |
