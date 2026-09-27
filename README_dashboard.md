# ClimaGuard — Flask dashboard

A regional **environmental health risk nowcast** for Bangladesh. It is **not an individual diagnosis**.

## Run

```bash
pip install -r requirements.txt
python scripts/build_page_assets.py   # once (~4 min): precomputes artifacts/*.json
python app.py                         # http://127.0.0.1:5000
python -m pytest -q                   # offline (Open-Meteo is faked)
```

`scripts/build_page_assets.py --only dataset eda ...` rebuilds only some assets. The slow part is `shap`, about 3 minutes for the tree models.

On first start the app loads the six models (~2–8 s). It also fetches Dhaka's 2015–2025 climatology from the Open-Meteo archive once and caches it in `data/cache/dhaka_climatology.json`. A background thread then:
- fetches the live data;
- runs the models;
- builds the SHAP explainers;
- loads the dataset rows used by the map drawer.

After that, page loads don't wait on any of these. When cached data goes stale it is served immediately while a refresh runs in the background (stale-while-revalidate). The refresh interval comes from Settings.

The legacy Streamlit app in `app/` is untouched.

## Pages

Every page shares the sidebar and header:
- The header has a division selector kept in `?loc=`, so reloading keeps the choice.
- Each page shows skeleton loaders while loading, an error card with Retry on failure, and a "Using last available data" banner in fallback mode.
- Below 1280px the sidebar collapses to icons.
- Light and dark themes use CSS variables; the Low/Moderate/High colours are the same in both.

| Route | Content | Real vs demo |
|---|---|---|
| `/` | Dashboard: hero score, 5 disease cards, SHAP bars, environment, 7-snapshot trend, advisories, what changed, mini map, profile strip | Scores, SHAP, environment and advisories are live and model-backed. Trend and "vs last…" show a Demo chip while backfilled weeks are in view |
| `/my-risk` | Per-disease tabs:<br>• model score → profile rules → final score<br>• local SHAP (top 8, signed)<br>• test R² and reliability badge<br>• advisory actions<br>• 8-snapshot sparkline | All model-backed. The sparkline shows a Demo chip while backfilled |
| `/environment` | 10 live tiles tagged "Used by model" or "Display only"; 7-day forecast; 72-h PM2.5/AQI with WHO/EPA lines; heat-wave watch | Live Open-Meteo. Status colours use real-world scales (EPA, WHO, NOAA, BMD), not the model |
| `/risk-map` | Division choropleth (live) or 25-country choropleth (dataset test weeks, with a week slider); layer selector; side drawer with 5 scores, top 3 SHAP drivers and advisories | Division view is marked Demo (country-level model). Country view is model output on real dataset weeks |
| `/risk-history` | Range selector; overall + 5 disease scores over time with risk bands; predicted vs actual on the test period; month × disease heatmap; last 12 snapshots with CSV download | Dataset series are model runs on real Bangladesh weeks. Snapshots show a Demo chip while backfilled |
| `/my-health` | Profile form (BMI can be computed from height and weight) with a live preview of every rule and its points | Saved to the SQLite `profile` table. The profile is never sent to the ML model |
| `/how-it-works` | 12 sections with a sticky nav: scope, dataset, cleaning, EDA, features, split, classifier comparison, regressors, global SHAP, advisory table, limitations, team | Every number comes from `artifacts/*.json` |
| `/settings` | °C/°F, light/dark, default location, refresh interval (drives the cache), notifications (UI only), About | Saved to the SQLite `settings` table |

## API

`loc` is one of: Dhaka, Chattogram, Rajshahi, Khulna, Sylhet, Barisal, Rangpur, Mymensingh. It defaults to the Settings default location; `location=` is accepted as an alias.

| Endpoint | Returns |
|---|---|
| `GET /api/dashboard?loc=` | The Dashboard payload: `overall`, `diseases[]`, `shap_factors[]`, `environment`, `trend`, `advisories[]`, `changes`, `map[]`, `updated_at` |
| `POST /api/recalculate` | Clears the cache, refetches Open-Meteo, re-runs inference. Same schema |
| `GET /api/disease/<key>?loc=` | My Risk tab data (key: respiratory, vector, heat, waterborne, cardio) |
| `GET /api/environment?loc=` | Environment tiles with status, meaning and the model's 7-day value |
| `GET /api/forecast?loc=` | 7-day daily forecast, 72-h air quality, heat-wave watch. Returns 502 if Open-Meteo is down |
| `GET /api/map?view=division\|country&layer=&week=` | Choropleth values (layer: overall or a disease; `week` indexes the test weeks) |
| `GET /api/map/detail?view=&id=&week=` | Drawer: 5 scores, top 3 SHAP drivers, advisories |
| `GET /api/history?loc=&range=4w\|12w\|1y\|all` | Dataset score series + the last 12 snapshots |
| `GET /api/history.csv?loc=` | Last 12 snapshots as CSV |
| `GET /api/test-predictions?target=` | Predicted vs actual on the test split |
| `GET /api/seasonality?scope=bgd\|all` | Month × disease average percentile |
| `GET /api/methodology` | Dataset stats, cleaning counts, correlations, EDA, features, metrics, R², global SHAP, advisory table |
| `GET/POST /api/profile`, `POST /api/profile/preview` | Profile (SQLite) and the live adjustment preview (does not save) |
| `GET/POST /api/settings`, `GET /api/about` | Settings (SQLite) and the About card |

Once warm, live endpoints take about 0.2–0.4 s. `recalculate` spends 2–4 s waiting on Open-Meteo.

## Models

The dashboard uses the saved Phase 6 validation winners from `models/phase6/`. Nothing was retrained.

| Task | Model | Test result |
|---|---|---|
| Overall class | Logistic Regression | from `output/phase6_metrics.csv` |
| Respiratory, vector, waterborne | Random Forest | ″ |
| Heat | XGBoost | ″ |
| Cardio | ElasticNet | R² ≈ 0 → "Low reliability" everywhere |

## How every number is computed

| Value | Source |
|---|---|
| Overall score | `100 × (0.5·P(Medium) + P(High))` from the classifier. Badge = argmax class ("Medium" is shown as "Moderate") |
| Disease scores | The regressor's prediction as a percentile of that target's training distribution (≤ 2022). <40 Low, 40–64 Moderate, ≥65 High |
| Changes | Points difference against the previous snapshot. The label says what that was: "vs yesterday", "vs last snapshot" or "vs last week (dataset)" |
| Profile adjustment | Rule-based, applied after the model, capped at ±10 per disease:<br>• asthma → resp +5<br>• high outdoor exposure → resp +5, heat +3<br>• age ≥ 65 → heat +5, cardio +5<br>• smoker → resp +3, cardio +3<br>• cardiovascular disease → cardio +5<br>• diabetes → cardio +3, heat +2<br>• BMI ≥ 30 → heat +3, cardio +2 |
| Local SHAP | Exact linear SHAP for the linear models; TreeExplainer for RF/XGBoost. Engineered columns are grouped under friendly labels. ⟲ marks autoregressive features (past values of a disease indicator) |
| Global SHAP (How It Works) | Mean \|SHAP\| of each saved model on 300 test weeks, recomputed by `build_page_assets.py`. The old `output/shap/*.png` files may describe other models |
| Advisories | `pipeline/step6_advisory.py` rules and thresholds (a disease rule fires at its 80th training percentile). Cardio is deliberately excluded |

### Building the live feature row

1. **Weather.** Open-Meteo data (historical-forecast API for daily weather, air-quality API for PM2.5/AQI) is aggregated into 13 seven-day blocks ending today. Lags, rolling windows and interactions are derived with the same rules as `pipeline/step3_feature_engineering.py`; a unit test checks the results match the dataset exactly.
2. **Temperature quantile mapping.** The dataset's `temperature_celsius` is not real °C (Bangladesh averages 11 °C, the UK −7.9 °C). A live weekly mean is placed at its percentile in Dhaka's real 2015–2025 weekly climatology, then mapped to the same percentile of the dataset's Bangladesh temperatures.
3. **Heat-wave days.** A day counts when its max temperature is above the 95th percentile of Dhaka's 2015–2024 daily maxima.
4. **Country context and disease lags** come from the last dataset week (2025-10-19). No live source exists for them.
5. **Missing values** are filled with the training median, as in step 5.

## Known gaps

- **Cardio has no predictive skill** (test R² ≈ 0). The UI flags it on every page.
- **Heat admissions are zero-inflated**, so heat scores jump between about 0 and 50+.
- **Disease-lag inputs are frozen** at 2025-10-19.
- **Division differences come only from local weather.** Every division uses Bangladesh's country context.
- **Backfilled history** is weekly dataset data, not daily. It gets replaced as real snapshots accumulate.
- **The Risk History ranges** apply to the dataset period (up to 2025-10-19). Live snapshots appear in the snapshots table.
- **Notifications are UI only.** The only effect is the bell dot.
- **Tailwind comes from the Play CDN** (fine locally, not for production).
