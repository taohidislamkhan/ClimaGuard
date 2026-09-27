# Build all sidebar pages for WeatherHealth AI (phase 2)

## 0. Before you start
1. Re-read `claude_code_dashboard_prompt.md`. All of its rules still apply here, especially the design tokens, the Low/Moderate/High colors, the "Demo Data" chips, honest data sourcing, and the disclaimer.
2. Look at the existing Dashboard page (`templates/dashboard.html`, `static/js/*`, `dashboard/*.py`). Reuse the same components everywhere: card, badge, sparkline, bar list, gauge, chip, and pill tabs.
   - Move shared pieces into reusable Jinja macros (`templates/components/*.html`) and shared JS modules instead of copy-pasting them.
3. First, fix the issues listed in §1. Then show me a short plan for all 7 pages (routes, new API endpoints, what data each widget uses) and wait for my OK.

## 1. Fix these on the existing Dashboard first
1. **Hero card overlap:** the disclaimer text ("This is an environmental risk estimate, not a medical diagnosis.") overlaps with "Last updated". Stack them properly with spacing, and make sure it works at 1280–1536px widths.
2. **"Why is your risk elevated?" bars:** the bars currently render as tiny dots. Render full horizontal bars (track + filled bar) like the reference image.
3. **"vs yesterday" percentages are unrealistic** (e.g. 83%, 61%, 52%). This happens because the history snapshots are weekly dataset rows, not days.
   - Label these values honestly, e.g. "vs last snapshot" / "vs last week", until real daily history exists.
   - Show the change as **points difference** (e.g. "↑ 6 pts"), not as a percent of a small number.
4. **Disease card overflow:** card titles and info icons get cut off at the right edge (e.g. "Cardiovascular Risk ⓘ"). Allow wrapping or shrink the font; nothing may clip.
5. **Feature label for lagged targets:** if a SHAP factor is a lag or rolling window of a target (e.g. "Recent waterborne cases"), keep it, but add a tooltip: "past values of this indicator (autoregressive feature)".

## 2. Global page shell (applies to all pages)
- Same sidebar. The active item is highlighted based on the current route.
- Same header on every page (location, date, bell, profile menu). Add a page title and a one-line subtitle under the greeting area. The greeting itself appears only on Dashboard.
- Routes:
  - `/` Dashboard
  - `/my-risk`
  - `/environment`
  - `/risk-map`
  - `/risk-history`
  - `/my-health`
  - `/how-it-works`
  - `/settings`
- **Location selector** in the header: a dropdown of the 8 Bangladesh divisions. It is shared by all pages and kept in a URL query param `?loc=`, so reloading the page keeps the selection.
- Every page shows skeleton loaders while data loads, an error card with a Retry button on failure, and a "Using last available data" banner when the live API falls back.

## 3. Page specs

### 3.1 My Risk (`/my-risk`) — deep dive per disease
- **Top:** overall gauge (smaller than on Dashboard) + badge + a short plain-language summary sentence.
- **Tabs:** one tab per disease (Respiratory, Vector-borne, Heat-related, Waterborne, Cardiovascular). Each tab contains:
  - Score, band, **model score before profile adjustment**, the profile adjustment (+/− points, with a list of the rules that fired), and the final score.
  - Local SHAP waterfall-style bar chart for this disease model: top 8 features, positive values in red, negative values in green, friendly labels.
  - The model's reliability for this disease: test R² from the evaluation results, plus a confidence badge:
    - R² ≥ 0.8 → "High reliability"
    - R² from 0.4 to 0.8 → "Moderate"
    - R² < 0.4 → "Low reliability — interpret with caution"
    
    Cardiovascular (R² 0.118) must clearly show the low-reliability warning.
  - The advisory actions for this disease from the advisory engine.
  - An 8-week sparkline of this disease score.

### 3.2 Environment (`/environment`) — current conditions and forecast
- **Current conditions:** large tiles for temperature, feels-like, humidity, rainfall, wind, UV, PM2.5, PM10, AQI, each with a status color and a one-line meaning (e.g. "PM2.5 78 µg/m³ — Unhealthy for sensitive groups").
- **Model input marking:** a small tag on every tile, either "Used by model" or "Display only".
  - Model inputs: temperature, precipitation, PM2.5, AQI, heat-wave indicator.
  - Display only: humidity, wind, UV, PM10.
- **Charts (Chart.js):**
  - Next 7 days: temperature max/min band + rainfall bars (Open-Meteo daily forecast).
  - Next 72 hours: PM2.5 and AQI line chart with WHO/US-EPA threshold lines.
- **Heat-wave watch card:** count of forecast days above the heat threshold used in the pipeline.

### 3.3 Risk Map (`/risk-map`) — full-screen map
- Leaflet map of Bangladesh, filling the page area, with division polygons colored as a choropleth by risk (Low/Moderate/High colors).
- **Top-left floating panel:**
  - Layer selector: Overall / Respiratory / Vector-borne / Heat / Waterborne / Cardiovascular.
  - Toggle between "Division view (live, demo)" and "Country view (dataset, 25 countries)". The country view is a world map choropleth of the model's predictions for the 25 dataset countries, with a week slider covering the test period (2024–2025).
- Clicking a division or country opens a side drawer with: its 5 disease scores, top 3 SHAP drivers, and advisories.
- Legend bottom-right. "Demo Data" chip on the division view with a tooltip: "Model trained on country-level data; division values use live local weather with country context."

### 3.4 Risk History (`/risk-history`)
- **Range selector:** 4 weeks / 12 weeks / 1 year / All (dataset period).
- **Main chart:** multi-line chart of overall + 5 disease scores over time. Series can be toggled via the legend. Low/Moderate/High bands are drawn as faint horizontal background zones.
- **Model check chart:** predicted vs actual for the selected disease on the **test period only** (2024–2025), from the dataset. This demonstrates that the model works on unseen data.
- **Seasonality heatmap:** month × disease, showing the average risk (matches the EDA finding: cardio peaks in winter, heat in summer, vector/waterborne in summer–autumn).
- A table of the last 12 snapshots with a CSV download button.

### 3.5 My Health (`/my-health`) — profile
- A form with the fields: name, age, location (division), BMI (or height + weight → auto-computed), activity level, outdoor exposure, smoking, and conditions (asthma, cardiovascular disease, diabetes — checkboxes).
- **Right panel "How your profile affects your scores":** a live preview that recalculates the adjustment as the user edits the form, showing each rule and its +/− points (max ±10 per disease).
- A clear note: "Your profile is never sent to the ML model. It only applies small transparent rule-based adjustments."
- Save the profile to the Flask backend (a SQLite `profile` table, single user). No browser storage.

### 3.6 How It Works (`/how-it-works`) — methodology page (this doubles as the presentation backup, so make it thorough)
Build it as a scrollable page with a sticky section nav on the left. Sections:
1. **Problem & scope:** regional nowcast, not a diagnosis.
2. **Dataset:** 14,100 weekly rows, 25 countries, 8 regions, 2015-01-04 to 2025-10-19, 30 raw columns. Zero nulls and zero duplicates. Show these as stat tiles.
3. **Cleaning:**
   - 374 negative AQI values clamped to 0
   - healthcare-access scores clipped to 100
   - disease rates and counts clipped to ≥0
4. **EDA:**
   - An interactive correlation heatmap of the key variables.
   - A scatter plot of temperature vs heat admissions, showing the non-linear spike above ~25°C.
   - Seasonal line charts.
   - Load all of this from the processed data. Do not hardcode the numbers.
5. **Feature engineering:**
   - A diagram of rolling windows (4/8/12 weeks), lags (1/2/4/8 weeks), sin/cos encodings, interaction terms, and dummies.
   - The funnel "160–262 features → ~60 selected" with the four selection methods.
6. **Chronological split:** a timeline graphic (train ≤2022, validation 2023, test 2024–2025) with a note explaining why a random split would leak future information.
7. **Model comparison:** table + grouped bar chart of Accuracy, Macro-F1 and ROC-AUC for Random Forest, Logistic Regression, XGBoost and Decision Tree. Highlight the best model. Load the values from the saved evaluation results file.
8. **Disease regressors:** R² per target, with the interpretation text from the project context. Include the honest negative result for cardiovascular.
9. **Explainability:** global SHAP bar chart per disease (images or interactive bars generated from the saved SHAP values).
10. **Advisory engine:** a table of risk level → actions per disease.
11. **Limitations (red-bordered box):**
    - no humidity, elevation or case counts in the training data
    - trained at country level
    - weekly data
    - prototype only, not a clinical tool
12. **Team:** Phoenix Force — Sayma Talukder Rupa, Md. Taohid Islam Khan Tazim, Farhan Tariq Jamee. Course: Data Analytics Laboratory, Summer 2026, UIU.

### 3.7 Settings (`/settings`)
- **Units:** °C / °F.
- **Theme:** Light / Dark. Implement dark mode with the CSS variables, applied across all pages.
- **Default location.**
- **Data refresh interval:** 15 / 30 / 60 minutes (default 30).
- **Notification preferences:** UI only, e.g. "Notify when any disease is High".
- **About card:** app version, model versions, dataset name, last model training date, and the disclaimer.
- Save the settings to the backend (SQLite `settings` table).

## 4. Backend additions
- Suggested new endpoints (adjust as needed; keep the existing schema style):
  - `/api/disease/<name>`
  - `/api/environment?loc=`
  - `/api/forecast?loc=`
  - `/api/map?layer=&view=&week=`
  - `/api/history?range=`
  - `/api/test-predictions?target=`
  - `/api/methodology` (bundles the dataset stats, cleaning counts, correlation matrix, model metrics, R² values and global SHAP)
  - `/api/profile`
  - `/api/settings`
- Precompute the heavy methodology assets (correlation matrix, test predictions, global SHAP) once into `artifacts/*.json` with a script `scripts/build_page_assets.py`, so the pages load fast.
- Keep the <800 ms target for the live inference endpoints.

## 5. Rules
- No hardcoded metric numbers in the templates. Everything comes from the artifacts or the API.
- Every value that is not model-backed gets a "Demo Data" chip or a source tooltip.
- The same Low/Moderate/High colors are used on every page.
- The disclaimer appears on My Risk, Risk Map and every advisory block.
- Responsive from 1280px to 1920px. The sidebar collapses to icons only below 1280px.

## 6. Workflow and acceptance
1. Fix §1 → show me a screenshot of the Dashboard.
2. Plan all the pages → wait for my OK.
3. Build the pages one at a time in this order:
   1. How It Works
   2. My Risk
   3. Risk Map
   4. Risk History
   5. Environment
   6. My Health
   7. Settings
4. After each page:
   - run `pytest`
   - open the page in the browser (Playwright) at 1536px width and take a screenshot
   - check the layout for overlap, clipping and empty states
   - fix any problems before moving on
5. Final summary: routes list, what is real vs demo on each page, and known gaps.
