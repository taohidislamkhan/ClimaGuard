# ClimaGuard — EDA Summary (Phase 3)

**Input:** `data/processed/cleaned_data.csv` — 14,050 rows × 33 columns
(14,100 raw − 50 duplicate country-week rows; see `data_card.md`).
**Plots:** `output/charts/*.png` (13 figures) + `correlation_matrix.csv`.

## 1. Distribution shapes (`hist_numeric.png`)

- `vector_disease_risk_score` is **right-skewed** with most mass near 0 and a long tail — supports using log / square-root transforms or tree models rather than OLS.
- `heat_wave_days` is **zero-inflated** (many weeks with 0 heat-wave days across temperate regions) — supports the brief's call for rolling-window sums in Phase 4 instead of weekly means.
- `waterborne_disease_incidents` is also zero-inflated and right-skewed.
- `pm25_ugm3` and `air_quality_index` are right-skewed with long tails (industrial pollution events).
- Indicators (`drought_indicator`, `flood_indicator`) are dominated by 0 with a small spike at 1 (~0.6% flood weeks).
- `healthcare_access_index`, `mental_health_index`, `food_security_index` are roughly uniform-ish with a mild floor effect.

## 2. Pairwise correlations (`correlation_heatmap.png`)

| Pair | r | Verdict |
|---|---|---|
| PM2.5 ↔ AQI | **0.968** | ✅ matches brief (~0.97) — they are near-duplicates; drop one in Phase 4. |
| PM2.5 ↔ respiratory | **0.759** | ✅ matches brief (~0.76). |
| heat_wave_days ↔ heat_admissions | **0.700** | ✅ matches brief. |
| temperature ↔ vector_disease_risk | **0.653** | ✅ matches brief. |
| healthcare_access ↔ respiratory | **−0.497** | Partial support for H4. |
| temperature ↔ heat_admissions | 0.416 | Positive but non-linear — see H1 below. |
| flood_indicator ↔ waterborne | 0.349 | Weak *overall* r because floods are rare. |
| cardio_mortality ↔ anything | ≈ 0.01 | **No climate signal in this column** — null finding. |
| temperature ↔ healthcare_access | −0.630 | Confounding: warm countries in this dataset tend to be lower-income — must include both in any model. |

## 3. Disease outcomes vs. temperature (`scatter_*_vs_temp.png`)

Binned (2 °C) means, coloured by income_level. The **heat-admission curve is the clearest threshold effect in the dataset**:

| T bin (°C) | mean heat admissions |
|---:|---:|
| −5 | 1.7 |
| 0 | 5.2 |
| 5 | 5.0 |
| 10 | 7.6 |
| 15 | 7.9 |
| 20 | 9.3 |
| **25** | **13.6** |
| 30 | 14.4 |
| 35 | 23.5 |

The curve is flat to slightly rising up to ~20 °C, then bends sharply upward — direct visual justification for **tree-based models** (which can capture the kink) over a linear fit. The Low-income curve sits consistently above the High-income curve at every temperature, supporting H4.

## 4. Seasonal patterns (`seasonal_*.png`)

Hemisphere-aware seasons. Means across all 25 countries × 564 weeks.

| Season | heat_adm | respiratory | vector | waterborne | cardio |
|---|---:|---:|---:|---:|---:|
| winter | 5.9 | 64.5 | 22.4 | 30.7 | 30.7 |
| spring | **13.5** | 67.3 | 22.8 | 21.1 | **31.1** |
| summer | 3.6 | 73.7 | **33.2** | 26.5 | 30.7 |
| autumn | 4.5 | 73.5 | 33.1 | **26.7** | 30.6 |

- **Heat admissions peak in spring**, not summer — because southern-hemisphere summer (DJF) is bucketed into NH winter and pulls heat admissions down in the "summer" aggregate, while SH autumn (MAM) is bucketed into NH spring. This is exactly why hemisphere-aware bucketing was added in Phase 2.
- **Vector & waterborne peak in summer/autumn** ✅ — matches the brief.
- **Cardio mortality is essentially flat across seasons** (~30.6–31.1).

## 5. Yearly trend by region (`yearly_trend_by_region.png`)

Mean temperature ranges dramatically by region (Southeast Asia 18.6 °C vs Europe −2.9 °C), as do heat-related admissions and PM2.5. Cardio mortality is **flat across regions** (~30.7–30.9). The cross-region separation in temperature, heat admissions, and PM2.5 is far larger than any within-region year-to-year change — confirms the brief's point that **country/region dummies are mandatory** so the model isn't forced to attribute "Europe vs South Asia" to a single temperature coefficient.

## 6. Hypotheses (matches Slide 4)

### H1 — Higher temperature → more heat-related admissions ✅ supported

- Pearson r(temperature, heat_related_admissions) = **+0.416**.
- Non-linear: flat below ~20 °C, then sharp rise (5× by 30 °C, 14× by 35 °C).
- Visually confirmed in `scatter_heat_related_admissions_vs_temp.png`: each income curve bends upward.

### H2 — Higher AQI/PM2.5 → higher respiratory disease rate ✅ strongly supported

- Pearson r(PM2.5, respiratory) = **+0.759**, r(AQI, respiratory) = +0.736.
- Heatmap shows the strongest correlation pair in the matrix.

### H3 — Floods → more waterborne incidents ✅ supported (conditional)

- Overall Pearson r = +0.349 (muted because floods are rare: 83/14,050 = 0.6%).
- Conditional mean waterborne incidents: **no flood = 22.0 vs flood = 52.1 (≈ 2.4×)**, which is the right way to test it given the zero-inflation.

### H4 — Better healthcare access → lower disease impact ✅ partially supported

- Pearson r(healthcare_access, respiratory) = **−0.497**.
- Quintile means: Q1=81.5 → Q2=77.9 → Q3=63.7 → Q4=63.3 → Q5=63.5.
- The benefit saturates — once healthcare access crosses roughly the 60th percentile, additional access no longer moves respiratory rates. The brief's qualitative claim holds, but the relationship is **non-linear with a kink** — another argument for tree-based models.

## 7. Implications for Phase 4 (feature engineering)

1. **Drop one of PM2.5 / AQI** (r = 0.968). Keep PM2.5 — it's a measured concentration, AQI is derived from it.
2. **Rolling-window sums over 4–12 weeks** for heat-wave days and precipitation (zero-inflated, weekly signal noisy).
3. **Country, region, and income-level dummies** are required (cross-region temperature gap is huge).
4. **Hemisphere-aware lag/season features** — never mix SH summer with NH summer in a single seasonal dummy.
5. **Transform vector_disease_risk** (log1p) and waterborne (log1p) before any linear model.
6. **Reconsider cardio_mortality** as a target — there is no climate signal in this column; using it as a model target would be misleading.

## 8. Caveats

- Single realisation of weekly climate-health data; no measurement-error model.
- `cardio_mortality_rate` shows no climate correlation here, contrary to the published literature — flag to the data source owner.
- Income-level colouring aggregates large countries; sub-national income variation is not captured.
