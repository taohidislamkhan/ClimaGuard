# ClimaGuard

Climate-driven public health impact tracker and advisory system (2015–2025).

## Structure

```
ClimaGuard/
├── data/
│   ├── raw/global_climate_health_impact_tracker_2015_2025.csv
│   └── processed/cleaned_data.csv
├── pipeline/
│   ├── step1_clean_inspect.py
│   ├── step2_eda.py
│   ├── step3_feature_engineering.py
│   ├── step4_feature_selection.py
│   ├── step5_train_model.py
│   └── step6_advisory.py
├── output/            # charts, tables, model files
├── models/             # serialized .pkl / .json
├── app/
│   └── advisory_app.py  # Streamlit dashboard
├── run_all.py
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
streamlit run app/advisory_app.py
```

## Pipeline

1. **Clean & inspect** — handle missing values and duplicates.
2. **EDA** — histograms and country rankings.
3. **Feature engineering** — anomalies, interactions, encodings.
4. **Feature selection** — correlation-based pruning.
5. **Train model** — RandomForest regressor, persisted to `models/`.
6. **Advisory** — rule-based alerts from the predicted impact score.
