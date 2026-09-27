# Build the "WeatherHealth AI" dashboard — exact replica of the reference design

## 0. Before you write any code
1. Look at the reference screenshot: `docs/reference_dashboard.png`. This is the target. Match it as closely as possible: layout, spacing, colors, typography, card shapes, icons, and chart styles.
2. Explore the repo first and tell me what you found:
   - trained models (e.g. `models/*.pkl` / `*.json`): overall risk classifier (Random Forest, Low/Medium/High) and 5 XGBoost regressors (respiratory_disease_rate, vector_disease_risk_score, heat_related_admissions, waterborne_disease_incidents, cardio_mortality_rate)
   - processed data / engineered features, the selected feature list, SHAP artifacts
   - the rule-based advisory engine
   - whether **Bangladesh** is one of the 25 countries in `global_climate_health_impact_tracker_2015_2025.csv`
3. Then show me a short implementation plan (files you will create, API endpoints, how each UI number is computed) and wait for my OK before building.

## 1. Project context (do not change the science)
- University Data Analytics Lab project: ML system that predicts **regional** disease risk (Low/Medium/High) from weather, air quality, location and socioeconomic data. It is a **regional nowcast, NOT an individual diagnosis**.
- Dataset: 14,100 weekly rows, 25 countries, 2015–2025. It has NO humidity, NO UV index, NO wind speed, NO elevation, and NO sub-national (division/city) data.
- Models were trained on a chronological split. Do not retrain or modify them unless a model file is missing. If something is missing, stop and tell me.

## 2. Tech stack
- Backend: **Flask** (Python 3.11), serving JSON endpoints + one HTML page. Load models once at startup.
- Frontend: single page, **HTML + Tailwind CSS (CDN) + vanilla JS**.
  - Charts: Chart.js (gauge, sparklines, 7-day trend)
  - Map: Leaflet with a Bangladesh divisions GeoJSON (store it in `static/geo/`)
  - Icons: Lucide
  - Font: Inter (Google Fonts)
- Live weather: **Open-Meteo** forecast + air-quality APIs (free, no key). Cache responses for 30 minutes.
- Everything must run locally with `pip install -r requirements.txt && python app.py`.

## 3. Design tokens (use CSS variables)
- Sidebar: dark navy gradient (`#0B2545` → `#13315C`), white text; active item uses a lighter blue pill (`#1B4F8A`) with a white icon.
- Page background: `#F4F7FB`. Cards: white, `border-radius: 12px`, a subtle 1px border `#E6ECF3`, and a soft shadow.
- Primary button: teal-blue `#1597BB` (Recalculate Risk); secondary dark-blue buttons `#0E4C7A`.
- Risk semantics, used EVERYWHERE (badges, bars, map dots, gauge):
  - Low = green `#22A06B` (badge background `#E3F6EC`)
  - Moderate = amber `#F5A524` (badge background `#FFF3D6`)
  - High = red `#E5484D` (badge background `#FDE7E8`)
- Text: headings `#0F1B2D`, body `#475467`, muted `#8A94A6`. Small uppercase badge labels.
- Small "Demo Data ⓘ" / "Demo SHAP-style explanation ⓘ" chips: light blue pill, 10px text.

## 4. Layout (desktop 1536px wide, grid-based, responsive down to 1280px)

### Sidebar (fixed, ~250px)
- Logo + "WeatherHealth AI" / "Environmental Health Intelligence".
- Nav groups:
  - Group 1: Dashboard (active), My Risk, Environment, Risk Map, Risk History
  - Divider, then Group 2: My Health, How It Works
  - Divider, then Group 3: Settings
- Bottom: green dot "System Online", and "Data updated / Today, HH:MM PM".
- Only Dashboard needs full content now. The other pages are simple routes:
  - Risk Map = full-screen map
  - Risk History = trend page
  - How It Works = methodology, 4-model comparison table, SHAP global plots, and limitations
  - The rest can be placeholders.

### Header
- "Good evening, {name} 👋" (greeting changes by time of day), subtitle "Here's your environmental health overview for today."
- Right side: location pin + "Dhaka, Bangladesh" + date, a bell icon with a red dot, and an avatar circle + name dropdown.
- Below the header on the right: the **Recalculate Risk** button (refresh icon), which re-runs inference.

### Row 1 — Hero card (full width)
- Left: a circular gauge (amber arc proportional to score) showing "62 / 100", with the caption "Environmental Risk Score".
- Middle:
  - "Your Environmental Health Risk" + a risk badge ("MODERATE RISK")
  - "↑ 8% compared with yesterday" (red if up, green if down)
  - One explanatory line
  - "View Detailed Analysis →" button
- Top-right:
  - Info icon + "This is an environmental risk estimate, not a medical diagnosis."
  - "Last updated: HH:MM PM"
- Right background: a soft illustrated city skyline with trees and a sun (a simple SVG is fine, light blue/green tones, low opacity).

### Row 2 — Five disease cards (equal width)
- Respiratory, Vector-borne, Heat-related, Waterborne, Cardiovascular.
- Each card contains:
  - a colored icon (lungs, mosquito, thermometer-sun, droplet, heart-pulse)
  - the title and a large score number
  - a risk badge and "↑/↓ x% vs yesterday"
  - a small sparkline in the badge color, with a faint gradient fill

### Row 3 — Three cards
1. **Why is your risk elevated?**
   - Horizontal bars for the top 6 factors: value, colored bar, and a label ("High / Moderate / Low contribution").
   - Bars are colored red → orange → amber → teal → light teal.
   - Uses REAL SHAP values (see §5); keep the chip.
2. **Current Environment**
   - Icon grid: Temperature, Humidity, AQI (highlighted red when unhealthy), PM2.5, Rainfall, Wind Speed, UV Index.
   - Each tile has a small trend arrow.
   - An amber "Air Quality — Unhealthy for sensitive groups" callout box.
3. **Your 7-Day Risk Trend**
   - Pill tabs: Overall / Respiratory / Heat / Vector / Waterborne.
   - Line chart with points and value labels, Mon–Sun, y-axis 30–90, light area fill.

### Row 4 — Three cards
1. **Today's Advisories**
   - Stacked alert blocks, each with an icon, a title, a one-line reason, and 3 bullets:
     - red-tinted block for air quality
     - amber-tinted block for heat
   - Footer disclaimer: "General environmental health guidance. For medical concerns, consult a qualified healthcare professional."
   - Content comes from the existing rule-based advisory engine.
2. **What Changed Today?**
   - "Your risk increased by N points compared with yesterday."
   - "Main contributors" list with ↑/↓ percentages (PM2.5, Humidity, Temperature).
3. **Environmental Risk Map**
   - Leaflet map of Bangladesh with colored dots on the 8 divisions: Dhaka, Chattogram, Rajshahi, Khulna, Sylhet, Barisal, Rangpur, Mymensingh.
   - A side legend listing each division with its risk level in the matching color.
   - "Open Full Map →" button.

### Row 5 — "Your Health Profile" strip (full width)
- Six items with icons: Age, Location, BMI, Activity, Outdoor Exposure, Smoking.
- "Update Profile →" button opens a modal form; save the profile in JS state (no browser storage needed).

## 5. Data mapping — how every number is computed (be honest, no fake precision)
- **Disease scores (0–100):**
  1. Run each XGBoost regressor on the latest feature row.
  2. Convert the prediction to a percentile of that target's training distribution (0–100).
  3. Bands: <40 Low, 40–64 Moderate, ≥65 High.
- **Overall score:** `100 × (0.5·P(Medium) + 1.0·P(High))` from the Random Forest classifier's probabilities. The badge is the argmax class.
- **Personal adjustment (profile):** small, transparent, rule-based modifiers applied AFTER the model, capped at ±10 points. Examples:
  - asthma or high outdoor exposure → respiratory +5
  - age ≥ 65 → heat and cardio +5
  - smoker → respiratory +3
  
  Show a tooltip that explains the adjustment. Never feed the profile into the ML model.
- **"vs yesterday" and the 7-day trend:** compute from stored daily snapshots (a SQLite table `risk_history`). If history doesn't exist yet, backfill it by running the models on the last 7 weeks of the dataset, and show the "Demo Data" chip until real daily history accumulates.
- **Why is your risk elevated:** real local SHAP values from the relevant model (TreeExplainer) for the current row.
  - Map engineered feature names to friendly labels (e.g. `pm25_roll4_mean` → "PM2.5").
  - Normalize absolute SHAP values to 0–100 for bar length.
  - Only show features that exist in the model. **Humidity is NOT a model feature, so it must not appear here.** Replace it with the next real feature.
- **Current Environment:**
  - Temperature, rainfall, PM2.5 and AQI come from Open-Meteo live data (these match model inputs).
  - Humidity, wind speed and UV index are display-only live values from Open-Meteo. Put a small note saying they are not used by the model.
- **Bangladesh division map:**
  - The dataset has no division-level data.
  - For each division, pull live Open-Meteo weather/air quality at its lat/lon, build the feature row using the Bangladesh country context (or the closest South Asia country context if Bangladesh isn't in the dataset — tell me which), and run the models.
  - Keep the "Demo Data" chip and a tooltip saying the model was trained at country level.
- If Open-Meteo is unreachable, fall back to the latest dataset week and show a "Using last available data" banner.

## 6. API endpoints
- `GET /api/dashboard?location=Dhaka` → one JSON object with everything the page needs:
  - `overall`, `diseases[]`, `shap_factors[]`, `environment`, `trend{overall, respiratory, heat, vector, waterborne}`, `advisories[]`, `changes[]`, `map[]`, `updated_at`
- `POST /api/recalculate` → forces a cache refresh + inference, and returns the same schema.
- `GET/POST /api/profile`
- Target: <800 ms per inference call (excluding the first cold start).

## 7. File structure
```
app.py
dashboard/
  inference.py
  scoring.py
  shap_utils.py
  weather_client.py
  advisory.py        (wrap the existing engine)
  history.py
templates/
  base.html
  dashboard.html
  (other page templates)
static/
  css/app.css
  js/dashboard.js
  js/charts.js
  js/map.js
  geo/bd_divisions.geojson
  img/skyline.svg
tests/
  test_scoring.py
  test_api.py
requirements.txt
README_dashboard.md
```

## 8. Non-negotiable rules
- The disclaimer "This is an environmental risk estimate, not a medical diagnosis." must appear on the hero card and in advisories.
- Never hardcode the numbers from the screenshot (62, 72, 48…). They are only visual placeholders. All values come from the API.
- Every value that is not model-backed carries a "Demo Data" chip or a tooltip explaining its source.
- Colors for Low/Moderate/High must be identical across all components.

## 9. Workflow and acceptance
1. Plan → wait for my OK.
2. Build the backend and write tests (`pytest` must pass).
3. Build the frontend.
4. Run the app, take a screenshot of the page at 1536×1024 (e.g. with Playwright), and compare it side-by-side with `docs/reference_dashboard.png`. List the visible differences and fix them. Repeat until it closely matches.
5. Finish with a short summary: how to run it, what is real vs demo, and any known gaps.
