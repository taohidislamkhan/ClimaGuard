# ClimaGuard — Regional Environmental Disease Risk Intelligence

A Streamlit dashboard on top of a chronological-split ML pipeline
(25 countries × 564 weeks of climate + health data, 2015-01-04 → 2025-10-19).
It predicts **regional** Low/Medium/High environmental disease risk,
not individual medical diagnoses.

## Quick start

```powershell
# 1. Install
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. (Optional) regenerate pipeline artefacts
python run_all.py --skip-slow

# 3. Launch the dashboard
streamlit run app/app.py
# → open http://localhost:8501
```

## Architecture

```
ClimaGuard/
├── app/
│   ├── app.py                  # Entry point: page router + theme + sidebar
│   ├── components/
│   │   ├── theme.py            # CSS + colour palette + ring-score renderer
│   │   ├── sidebar.py          # Dark-navy nav + country/date inputs
│   │   ├── layout.py           # page_header, disease_card, shap_bars, env_cell
│   │   └── data.py             # @cache_data loaders (cleaned data, metrics, SHAP)
│   ├── pages/
│   │   ├── dashboard.py        # Landing screen (hero + 5 disease cards + …)
│   │   ├── my_risk.py          # Country + date picker → overall + per-disease
│   │   ├── environment.py      # Time-series of climate / air-quality for selection
│   │   ├── risk_map.py         # Full-screen Plotly choropleth + scrubber
│   │   ├── risk_history.py     # Multi-country P(High) trajectories
│   │   ├── regional.py         # Country & region comparison
│   │   ├── disease.py          # Per-disease model summary + SHAP top features
│   │   ├── models.py           # Classifier + per-disease R² tables
│   │   ├── explain.py          # Global / beeswarm / local SHAP
│   │   ├── methodology.py      # Dataset card, correlation heatmap, limitations
│   │   ├── how.py              # 8-step pipeline walkthrough
│   │   └── settings.py         # Diagnostics + artefact status
│   └── utils/
│       └── predict.py          # Inference layer: assess(), historical_*(), SHAP
├── pipeline/                   # step1 → step7 (clean, EDA, features, …)
├── models/phase6/              # trained .joblib winners
├── output/                     # advisories.csv, phase6_metrics.csv, shap_*.csv
├── data/processed/             # cleaned_data.csv
├── run_all.py                  # orchestrator (Phase 1-7)
├── requirements.txt
├── Dockerfile                  # python:3.11-slim + streamlit
└── .gitignore / .dockerignore
```

## Pipeline

```
step1  step2    step3                step4              step5              step6              step7
clean → EDA → 163 engineered → committee-vote → 4 models × 6 tasks → rule-based → SHAP
                features        (top-60 cap;        (chronological split) advisories       importance
                                 54 survive, 50 used)
```

### Phase 6 model results (test split, 2024-01-07 → 2025-10-19)

| Task            | Winner | Metric | Value |
|---|---|---|---|
| Overall classifier | Logistic Regression | macro-F1 / ROC-AUC | 0.819 / 0.944 |
| Respiratory        | Random Forest       | R² / RMSE            | 0.560 / 10.16 |
| Vector-borne       | Random Forest       | R² / RMSE            | 0.916 / 5.11 |
| Heat-related       | XGBoost             | R² / RMSE            | 0.830 / 4.04 |
| Waterborne         | Random Forest       | R² / RMSE            | 0.627 / 3.99 |
| Cardiovascular     | ElasticNet          | R² / RMSE            | **−0.008** / 5.63 |

> **Note.** The brief lists proposal numbers (RF acc 0.617 / F1 0.615 / ROC 0.799,
> cardio R² 0.118). The dashboard shows what the actual pipeline produced on the
> held-out test split — different but the same models and methodology.
>
> Winners are the models step5 picked on **validation** RMSE / macro-F1
> (`output/phase6_run_summary.json`) — the ones saved in `models/phase6/`.
> The app never re-picks a winner by test score.

## What the dashboard shows

| Page | Purpose |
|---|---|
| **🏠 Dashboard** | Landing: greeting + hero risk score (donut), 5 disease cards, why-elevated SHAP, current environment, 12-week trend, advisories, what-changed, mini map, regional profile |
| **📊 My Risk** | Country + date picker → overall risk + per-disease cards + P(High) history |
| **🌦️ Environment** | 52-week time series of temperature / anomaly / PM2.5 / AQI / precip / heat waves |
| **🗺️ Risk Map** | Choropleth with risk switcher (overall + 5 disease) + date scrubber |
| **📈 Risk History** | Multi-country P(High) comparison |
| **🌍 Regional Analysis** | Per-country + per-region ranking |
| **🦠 Disease Analysis** | 5 disease cards: R², model, predicted vs actual, SHAP features |
| **🤖 Model Performance** | Classifier + per-disease R² tables, grouped bar chart |
| **🔍 Explainability** | Global SHAP / beeswarm / local SHAP for the current query |
| **📊 Data & Methodology** | Dataset card, correlation heatmap, feature families, **prominent limitations box** |
| **❓ How It Works** | 8-step pipeline walkthrough |
| **⚙️ Settings** | Artefact status + diagnostics |

## Inference layer

`app/utils/predict.py` is the single source of truth for what the UI reads:

- `list_countries()` — country reference rows
- `assess(country_code, date)` — full RiskAssessment (overall class, P(High), per-disease
  predictions and their percentile within the country's 2015-2025 history)
- `historical_overall(country_code, n_weeks)` — recent P(High) trajectory
- `historical_disease(country_code, disease_key, n_weeks)` — recent actual and predicted disease value
- `regional_snapshot(date, level, task)` — per-country summary for the choropleth (overall or one disease)
- `what_changed(country_code, date)` — Δ between the requested week and the previous one
- `shap_global(task, top_k)` — top features by mean |SHAP|
- `shap_local(task, row)` — feature contribution for one prediction (exact linear SHAP
  for the scaled linear winners, TreeExplainer for RF / XGBoost)
- `classifier_metrics_test()`, `disease_metrics_test()`, `best_disease_model_r2()`

Everything is read from disk artefacts (`output/advisories.csv`,
`output/phase6_metrics.csv`, `output/shap_*.csv`, `models/phase6/*.joblib`)
or `data/processed/cleaned_data.csv` for the source rows that step6 didn't
persist (raw environment columns like `temp_anomaly_celsius`, and the five
actual disease values). Per-disease predictions aren't persisted by step6
either, so the app runs each saved disease winner on the advisory rows'
features once at startup (`pred_<target>` columns).

### Data integrity

- Every number on the dashboard is computed from real artefacts.
- Personal-factor toggles are not implemented — the source file has no
  individual health data. The "personalized" advisory variant from the
  original brief is replaced by a *regional* profile (country, region,
  income, climate, population).
- The cardiovascular model's R² is reported as a negative value on the
  Disease Analysis page — an honest negative result, not a bug.

## Caching

- `@st.cache_data` on every CSV loader (cleaned data, advisories, metrics,
  SHAP tables). The 14,050-row cleaned panel is loaded once per session.
- `@st.cache_resource` on the `.joblib` model loader. Models are loaded
  once and held in memory.
- Streamlit's standard `cache_data` warnings ("No runtime found, using
  MemoryCacheStorageManager") appear in the bare-mode test runs — they
  do **not** appear when the dashboard is launched normally.

## Performance

- First cold start: 5–10 s on commodity hardware (CSV + model loads).
- Subsequent renders: < 800 ms for any single dashboard query.
- All six winners are loaded once at startup, because the per-disease
  predictions are computed when the advisory table is first loaded.

## Deployment

### Streamlit Community Cloud (recommended)

```bash
git push              # .gitignore excludes venv, raw data, intermediate CSVs
# → share.streamlit.io → New app → point at app/app.py
```

### Docker

```bash
docker build -t climaguard .
docker run --rm -p 8501:8501 climaguard
```

### Render / Railway / Fly.io

Use the included `Dockerfile` — these services auto-detect it. Set the
service port to `8501`.

## Limitations

- **Prototype for academic / research use** — not a clinical diagnostic tool.
- Predicts **regional environmental risk**, not individual disease.
- Dataset does **not** contain humidity, wind speed, UV index, elevation,
  BMI, smoking, asthma history, or named per-disease case counts.
- Historical dataset is **weekly** — daily granularity is not modelled.
- Model performance varies substantially by disease target.
- **Cardiovascular model has negative R²** — climate is a weak predictor;
  other drivers dominate. Reported honestly.
- Predictions should **not** be interpreted as medical advice.

## Running the full pipeline from scratch

```bash
python run_all.py                # ~5 min end-to-end
python run_all.py --skip-slow    # skip step7 SHAP
python run_all.py --from-step 5  # resume from step 5
```

Determinism: every script pins `random_state=42`; step5/6/7 seed
Python + NumPy + framework RNGs. Two back-to-back runs produce identical
artefacts (SHA-256 verified across `output/*.csv/json` and `models/phase6/*.joblib`).
