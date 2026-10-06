"""Data for the sidebar pages (everything except the Dashboard payload).

Live, per-location data comes from the cached model run in
``DashboardService``; static assets (metrics, SHAP, dataset predictions)
come from ``artifacts/*.json`` built by the DVC ``assets`` stage, and model
health from ``reports/drift/drift_report.json`` (the ``drift`` stage).
"""

from __future__ import annotations

import csv
import io
import json
import threading
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import advisory, personal_rules, scoring, shap_utils, weather_client
from src.core.features import drop_first_week
from src.utils import paths
from src.utils.config import load_params

from .inference import FEATURED_PATH, MODEL_DIR, MODEL_NAMES, TARGETS, TASKS
from .service import DISEASE_LABELS, DISEASE_ORDER, DashboardService, aqi_category
from .storage import DEFAULT_PROFILE, clean_profile

ART = paths.ARTIFACTS
APP_VERSION = "2.0.0"
LAYERS = ["overall"] + DISEASE_ORDER
RANGES = {"4w": 4, "12w": 12, "1y": 52, "all": None}

INTERPRETATION = {
    "respiratory": "PM2.5 and the PM2.5 × AQI interaction dominate. About half of the week-to-week "
                   "variance is explained; the rest depends on factors not in the data (smoking rates, "
                   "indoor air, reporting).",
    "vector": "Temperature and rainfall × temperature drive mosquito-borne risk almost monotonically — "
              "the strongest model in the project.",
    "heat": "Season, temperature and heat-wave days explain most of the variance. The target is "
            "zero-inflated (about half of training weeks have no admissions).",
    "waterborne": "Past waterborne cases (autoregressive lags) and food security carry most of the "
                  "signal; weather adds a moderate contribution.",
    "cardio": "Honest negative result: climate features do not predict weekly cardiovascular mortality "
              "in this dataset. The model is no better than predicting the average, so its score "
              "should not be interpreted.",
}


class ArtifactMissing(RuntimeError):
    pass


def model_health() -> dict | None:
    """Summary of the drift stage's report, or None if it has not been run."""
    if not paths.DRIFT_REPORT.exists():
        return None
    r = json.loads(paths.DRIFT_REPORT.read_text())
    return {"retrain_recommended": r["retrain_recommended"], "reasons": r["reasons"],
            "response": r["response"], "data": r["data_drift"], "concept": r["concept_drift"],
            "thresholds": {k: r["thresholds"][k] for k in ("psi_threshold", "ks_pvalue", "f1_tolerance", "rmse_tolerance",
                                                           "max_drifted_share", "max_flagged_share")}}


@lru_cache(maxsize=None)
def artifact(name: str) -> dict:
    path = ART / name
    if not path.exists():
        raise ArtifactMissing(f"{path.name} not found — run `dvc pull` or `dvc repro assets`.")
    return json.loads(path.read_text())


def reliability(r2: float | None) -> dict:
    if r2 is None:
        return {"label": "Unknown", "level": "Moderate"}
    if r2 >= 0.8:
        return {"label": "High reliability", "level": "Low"}          # green
    if r2 >= 0.4:
        return {"label": "Moderate reliability", "level": "Moderate"}
    return {"label": "Low reliability — interpret with caution", "level": "High"}  # red


def test_r2(task: str) -> float | None:
    m = artifact("metrics.json")
    winner = m["winners"].get(task)
    for row in m["rows"]:
        if row["task"] == task and row["model"] == winner and row["split"] == "test":
            return row.get("r2")
    return None


# ---------------------------------------------------------------------------
# Environment tile status (real-world category scales, not model output)
# ---------------------------------------------------------------------------
def _band(value, cuts, labels):
    """cuts ascending; labels has len(cuts)+1 (level, meaning) pairs."""
    if value is None:
        return {"level": None, "meaning": "No data"}
    for cut, lab in zip(cuts, labels):
        if value < cut:
            return {"level": lab[0], "meaning": lab[1]}
    return {"level": labels[-1][0], "meaning": labels[-1][1]}


def tile_status(key: str, v: float | None, heat_tmax: float) -> dict:
    if key == "temperature":
        return _band(v, [32, heat_tmax], [("Low", "Normal range"), ("Moderate", "Hot"),
                                          ("High", f"Above the local heat-wave threshold ({heat_tmax:.0f} °C)")])
    if key == "feels_like":
        return _band(v, [32, 41], [("Low", "Comfortable to caution (NOAA heat index)"),
                                   ("Moderate", "Extreme caution (NOAA heat index)"),
                                   ("High", "Danger (NOAA heat index)")])
    if key == "humidity":
        return _band(v, [70, 86], [("Low", "Comfortable"), ("Moderate", "Humid"), ("High", "Very humid")])
    if key == "rainfall":
        return _band(v, [23, 89], [("Low", "Light to moderate rain (BMD scale)"),
                                   ("Moderate", "Heavy rain (BMD scale)"), ("High", "Very heavy rain (BMD scale)")])
    if key == "wind_speed":
        return _band(v, [20, 40], [("Low", "Light breeze"), ("Moderate", "Strong breeze"), ("High", "Gale risk")])
    if key == "uv_index":
        return _band(v, [3, 6, 8, 11], [("Low", "Low (WHO)"), ("Moderate", "Moderate (WHO)"),
                                        ("Moderate", "High (WHO)"), ("High", "Very high (WHO)"),
                                        ("High", "Extreme (WHO)")])
    if key == "pm25":
        return _band(v, [9.1, 35.5, 55.5, 125.5], [("Low", "Good (US EPA)"), ("Moderate", "Moderate (US EPA)"),
                                                   ("Moderate", "Unhealthy for sensitive groups (US EPA)"),
                                                   ("High", "Unhealthy (US EPA)"), ("High", "Very unhealthy (US EPA)")])
    if key == "pm10":
        return _band(v, [55, 155, 255, 355], [("Low", "Good (US EPA)"), ("Moderate", "Moderate (US EPA)"),
                                              ("Moderate", "Unhealthy for sensitive groups (US EPA)"),
                                              ("High", "Unhealthy (US EPA)"), ("High", "Very unhealthy (US EPA)")])
    if key == "aqi":
        c = aqi_category(v)
        return {"level": c["level"], "meaning": f"{c['label']} (US AQI)"}
    if key == "heat_wave_days":
        return _band(v, [1, 3], [("Low", "No heat-wave days this week"), ("Moderate", "Some heat-wave days"),
                                 ("High", "Sustained heat wave")])
    return {"level": None, "meaning": ""}


ENV_TILES = [
    # key, label, unit, used by model
    ("temperature", "Temperature", "°C", True),
    ("feels_like", "Feels like", "°C", False),
    ("humidity", "Humidity", "%", False),
    ("rainfall", "Rainfall (today)", "mm", True),
    ("wind_speed", "Wind", "km/h", False),
    ("uv_index", "UV index (max)", "", False),
    ("pm25", "PM2.5", "µg/m³", True),
    ("pm10", "PM10", "µg/m³", False),
    ("aqi", "AQI (US)", "", True),
    ("heat_wave_days", "Heat-wave days (7 d)", "days", True),
]


class PageData:
    def __init__(self, svc: DashboardService) -> None:
        self.svc = svc
        self._shap_cache: dict[tuple, list] = {}
        self._fd: pd.DataFrame | None = None
        self._flock = threading.Lock()

    # -- helpers ---------------------------------------------------------------
    def _run_bits(self, loc: str):
        run = self.svc.model_run()
        pred = run.preds[loc]
        return run, pred, self.svc.model_scores(pred)

    def _overall(self, pred, scores) -> dict:
        level = scoring.overall_level(pred.proba)
        return {"score": round(scores["overall"]), "level": level,
                "probabilities": {("Moderate" if k == "Medium" else k): round(v, 3)
                                  for k, v in pred.proba.items()}}

    def warm(self) -> None:
        """Build the tree explainers once so the first My Risk load is fast."""
        x = self.svc.store.to_matrix([self.svc.store.template])
        for key in DISEASE_ORDER:
            shap_utils.local_shap(self.svc.store.models[key], x)
        self._featured()          # dataset rows for the country drawer on the Risk Map

    # -- My Risk ---------------------------------------------------------------
    def disease(self, key: str, loc: str, profile: dict | None = None) -> dict:
        """Deep dive for one disease; ``profile`` is the signed-in user's saved one (or None)."""
        if key not in DISEASE_ORDER:
            raise KeyError(key)
        run, pred, scores = self._run_bits(loc)
        adjustments = scoring.personal_adjustments(profile) if profile else {}
        adj = adjustments.get(key, {"points": 0, "raw_points": 0, "capped": False, "reasons": []})
        final = scoring.apply_adjustment(scores[key], adj["points"])

        cache_key = (run.computed_at, loc, key)
        if cache_key not in self._shap_cache:
            self._shap_cache[cache_key] = shap_utils.grouped_signed(
                self.svc.store.models[key], run.rows[loc], k=8)
        shap_rows = self._shap_cache[cache_key]

        r2 = test_r2(key)
        rule = {"respiratory": "respiratory_high", "vector": "vector_high",
                "heat": "heat_high", "waterborne": "waterborne_high"}.get(key)
        if rule:
            q = advisory.THRESHOLDS[f"{key}_q"]
            actions = {"rule": rule, "fires": scores[key] >= q * 100, "threshold": round(q * 100),
                       "title": advisory.DISPLAY[rule]["title"],
                       "bullets": advisory._bullets(advisory.ADVISORY_RULES[rule])}
        else:
            actions = {"rule": None, "fires": False, "threshold": None, "title": "No advisory rule",
                       "bullets": [], "note": "The advisory engine deliberately excludes cardiovascular "
                                              "risk: the model has no predictive skill (R² ≈ 0)."}
        recent = self.svc.history.recent(loc, 8)
        return {
            "key": key, "label": DISEASE_LABELS[key], "location": loc, "live": run.live,
            "updated_at": run.computed_at.isoformat(timespec="seconds"),
            "overall": self._overall(pred, scores),
            "tabs": [{"key": k, "label": DISEASE_LABELS[k],
                      "score": round(scoring.apply_adjustment(
                          scores[k], adjustments.get(k, {"points": 0})["points"])),
                      } for k in DISEASE_ORDER],
            "model_score": round(scores[key]), "adjustment": adj, "score": round(final),
            "level": scoring.band(final), "model_level": scoring.band(scores[key]),
            "prediction": round(pred.diseases[key], 2), "target": TARGETS[key],
            "model": MODEL_NAMES[key],
            "shap": shap_rows,
            "shap_unit": f"contribution to predicted {TARGETS[key]} (target units)",
            "reliability": {"r2": r2, **reliability(r2)},
            "interpretation": INTERPRETATION[key],
            "advisory": actions,
            "sparkline": {"values": [round(r[key]) for r in recent],
                          "labels": [r["date"] for r in recent],
                          "sources": [r["source"] for r in recent]},
            "disclaimer": advisory.DISCLAIMER,
            "profile": profile,
            "personal": self._personal(key, loc, profile),
        }

    def _personal(self, key: str, loc: str, profile: dict | None) -> dict:
        """This disease's personal evaluation (rules fired, level, actions)."""
        pa = self.svc.personal_advisory(loc, profile)
        d = next(x for x in pa["diseases"] if x["disease"] == key)
        severe = personal_rules.LEVELS.index(d["personal_level"]) >= personal_rules.LEVELS.index("High")
        return {"personalized": pa["personalized"], **d,
                "red_flags": [d["red_flag"]] if severe else [], "disclaimer": pa["disclaimer"]}

    # -- Environment -----------------------------------------------------------
    def environment(self, loc: str) -> dict:
        run = self.svc.model_run()
        now, prev = run.env_now[loc], run.env_prev[loc]
        heat_tmax = self.svc.climatology().heatwave_tmax
        week = run.env_week[loc]
        hw = float(run.rows[loc]["heat_wave_days"].iloc[0]) if run.live else None
        values = {**now, "heat_wave_days": hw}
        tiles = []
        for key, label, unit, used in ENV_TILES:
            v = values.get(key)
            st = tile_status(key, v, heat_tmax)
            p = prev.get(key)
            tiles.append({
                "key": key, "label": label, "unit": unit, "value": v, "used_by_model": used,
                "level": st["level"], "meaning": st["meaning"],
                "trend": None if v is None or p is None else ("up" if v > p else "down" if v < p else "flat"),
                "model_value": {"temperature": week.get("temperature"), "rainfall": week.get("rainfall"),
                                "pm25": week.get("pm25"), "aqi": week.get("aqi"),
                                "heat_wave_days": hw}.get(key) if used else None,
            })
        return {"location": loc, "live": run.live, "observed_at": now.get("time"),
                "updated_at": run.computed_at.isoformat(timespec="seconds"), "tiles": tiles,
                "heatwave_tmax": heat_tmax}

    def forecast(self, loc: str) -> dict:
        f = weather_client.fetch_forecast(loc, self.svc.climatology().heatwave_tmax)
        return {"location": loc, **f,
                "thresholds": {"who_pm25_24h": 15, "aqi_usg": 100, "aqi_unhealthy": 150}}

    # -- Risk Map --------------------------------------------------------------
    def _layer_value(self, layer: str, pred, scores) -> tuple[int, str]:
        if layer == "overall":
            return round(scores["overall"]), scoring.overall_level(pred.proba)
        return round(scores[layer]), scoring.band(scores[layer])

    def map(self, view: str, layer: str, week: int | None) -> dict:
        if layer not in LAYERS:
            raise KeyError(layer)
        if view == "division":
            run = self.svc.model_run()
            items = []
            for name, (lat, lon) in weather_client.DIVISIONS.items():
                pred = run.preds[name]
                v, lvl = self._layer_value(layer, pred, self.svc.model_scores(pred))
                items.append({"id": name, "name": name, "lat": lat, "lon": lon, "value": v, "level": lvl})
            return {"view": view, "layer": layer, "live": run.live, "items": items,
                    "updated_at": run.computed_at.isoformat(timespec="seconds")}
        if view == "country":
            cm = artifact("country_map.json")
            weeks = cm["weeks"]
            i = len(weeks) - 1 if week is None else max(0, min(int(week), len(weeks) - 1))
            items = []
            for code, c in cm["countries"].items():
                v = c[layer][i]
                if v is None:
                    lvl = None
                elif layer == "overall":
                    lvl = c["level"][i]
                else:
                    lvl = scoring.band(v)
                items.append({"id": code, "name": c["name"], "value": v, "level": lvl})
            return {"view": view, "layer": layer, "week_index": i, "week": weeks[i],
                    "weeks": weeks, "items": items, "live": True}
        raise KeyError(view)

    def _featured(self) -> pd.DataFrame:
        with self._flock:
            if self._fd is None:
                self._fd = drop_first_week(pd.read_parquet(FEATURED_PATH))
            return self._fd

    def _detail(self, name: str, pred, x: pd.DataFrame) -> dict:
        scores = self.svc.model_scores(pred)
        level = scoring.overall_level(pred.proba)
        top = shap_utils.top_factors(self.svc.store.models["overall"], x, k=3)
        return {
            "name": name, "overall": self._overall(pred, scores),
            "diseases": [{"key": k, "label": DISEASE_LABELS[k], "score": round(scores[k]),
                          "level": scoring.band(scores[k])} for k in DISEASE_ORDER],
            "drivers": [{"label": f["label"], "direction": f["direction"], "value": f["value"],
                         "autoregressive": f["autoregressive"]} for f in top],
            "advisories": advisory.build_advisories("Medium" if level == "Moderate" else level,
                                                    pred.proba, scores, None),
            "disclaimer": advisory.DISCLAIMER,
        }

    def map_detail(self, view: str, item_id: str, week: int | None) -> dict:
        if view == "division":
            run = self.svc.model_run()
            if item_id not in run.preds:
                raise KeyError(item_id)
            return {"view": view, "source": "live Open-Meteo weather + Bangladesh country context",
                    **self._detail(item_id, run.preds[item_id], run.rows[item_id])}
        cm = artifact("country_map.json")
        if item_id not in cm["countries"]:
            raise KeyError(item_id)
        weeks = cm["weeks"]
        i = len(weeks) - 1 if week is None else max(0, min(int(week), len(weeks) - 1))
        name = cm["countries"][item_id]["name"]
        fd = self._featured()
        row = fd[(fd["country_name"] == name) & (fd["date"] == pd.Timestamp(weeks[i]))]
        if row.empty:
            raise KeyError(f"{item_id} {weeks[i]}")
        x = self.svc.store.to_matrix(row)
        pred = self.svc.store.predict(x)[0]
        return {"view": view, "week": weeks[i], "source": "dataset week (test period)",
                **self._detail(name, pred, x)}

    # -- Risk History ----------------------------------------------------------
    def history(self, loc: str, rng: str) -> dict:
        if rng not in RANGES:
            raise KeyError(rng)
        h = artifact("bgd_history.json")
        n = RANGES[rng]
        sl = slice(-n, None) if n else slice(None)
        series = {k: h[k][sl] for k in LAYERS}
        self.svc.ensure_backfill(loc)
        snaps = self.svc.history.recent(loc, 12)
        return {
            "range": rng, "location": loc,
            "dataset": {"dates": h["dates"][sl], "split": h["split"][sl], **series,
                        "source": "Model run on the dataset's Bangladesh weeks (country level)"},
            "snapshots": snaps,
            "demo": any(s["source"] == "backfill" for s in snaps),
        }

    def snapshots_csv(self, loc: str) -> str:
        rows = self.svc.history.recent(loc, 12)
        buf = io.StringIO()
        cols = ["location", "date", "source", "overall_level", "overall", "respiratory", "vector",
                "heat", "waterborne", "cardio", "pm25", "aqi", "temperature", "rainfall"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
        return buf.getvalue()

    # -- Methodology -----------------------------------------------------------
    def methodology(self) -> dict:
        metrics = artifact("metrics.json")
        tp = artifact("test_predictions.json")
        regressors = []
        for k in DISEASE_ORDER:
            r2 = test_r2(k)
            regressors.append({"key": k, "label": DISEASE_LABELS[k], "model": MODEL_NAMES[k],
                               "target": TARGETS[k], "r2": r2, "rmse": tp["targets"][k]["rmse"],
                               "reliability": reliability(r2), "interpretation": INTERPRETATION[k]})
        adv_table = [{"disease": DISEASE_LABELS[d], "trigger": f"score ≥ {round(advisory.THRESHOLDS[q] * 100)} "
                      f"(top {round((1 - advisory.THRESHOLDS[q]) * 100)}% of training weeks)",
                      "title": advisory.DISPLAY[rule]["title"],
                      "actions": advisory._bullets(advisory.ADVISORY_RULES[rule])}
                     for d, q, rule in advisory.DISEASE_RULES]
        adv_table += [{"disease": "Overall", "trigger": f"classifier predicts {lvl}",
                       "title": advisory.DISPLAY[rule]["title"],
                       "actions": advisory._bullets(advisory.ADVISORY_RULES[rule])}
                      for lvl, rule in [("Medium", "overall_medium"), ("High", "overall_high")]]
        split = load_params()["split"]
        adv_table.append({"disease": "Cardiovascular", "trigger": "none",
                          "title": "Excluded", "actions": ["No rule: the model has no predictive skill"]})
        return {
            "dataset": artifact("dataset.json"), "cleaning": artifact("cleaning.json"),
            "correlation": artifact("correlation.json"), "eda": artifact("eda.json"),
            "features": artifact("features.json"), "metrics": metrics,
            "regressors": regressors, "shap": artifact("shap_global.json"),
            "advisory_table": adv_table,
            "personal_engine": {
                "rules": personal_rules.rule_table(),
                "matrix": personal_rules.MATRIX,
                "overrides": personal_rules.CFG["overrides"],
                "heat_threshold": self.svc.climatology().heatwave_tmax,
                "sources": list(personal_rules.SOURCES.values()),
            },
            "split": {"train_end": split["train_end"], "val_end": split["val_end"],
                      "test_start": tp["start"], "test_end": tp["end"],
                      "sizes": {k: v["rows"] for k, v in json.loads(paths.SPLIT_SUMMARY.read_text()).items()}},
            "drift": model_health(),
            "disclaimer": advisory.DISCLAIMER,
        }

    def test_predictions(self, target: str) -> dict:
        tp = artifact("test_predictions.json")
        if target not in tp["targets"]:
            raise KeyError(target)
        return {"start": tp["start"], "end": tp["end"], "key": target,
                "label": DISEASE_LABELS[target], **tp["targets"][target]}

    def seasonality(self, scope: str) -> dict:
        s = artifact("seasonality.json")
        if scope not in s["values"]:
            raise KeyError(scope)
        return {"scope": scope, "months": s["months"], "metric": s["metric"],
                "rows": [{"key": k, "label": DISEASE_LABELS[k], "values": s["values"][scope][k]}
                         for k in DISEASE_ORDER]}

    # -- My Health ---------------------------------------------------------------
    def profile_preview(self, data: dict, loc: str, lang: str = "en",
                        base: dict | None = None) -> dict:
        """Preview ``data`` merged over ``base`` (the user's saved profile); nothing is saved."""
        p = clean_profile({k: v for k, v in data.items() if k != "name"}, base or DEFAULT_PROFILE)
        run, pred, scores = self._run_bits(loc)
        adj = scoring.personal_adjustments(p)
        return {
            "advisory": self.svc.personal_advisory(loc, p, lang),
            "profile": p, "rules": scoring.rule_table(p), "cap": scoring.ADJUST_CAP,
            "diseases": [{"key": k, "label": DISEASE_LABELS[k], "model_score": round(scores[k]),
                          "points": adj.get(k, {}).get("points", 0),
                          "raw_points": adj.get(k, {}).get("raw_points", 0),
                          "capped": adj.get(k, {}).get("capped", False),
                          "score": round(scoring.apply_adjustment(scores[k], adj.get(k, {}).get("points", 0))),
                          } for k in DISEASE_ORDER],
        }

    # -- Settings about card -------------------------------------------------------
    def about(self) -> dict:
        import sklearn
        import xgboost
        # File timestamps: when the model files were last written (the pipeline
        # does not record a training date inside the artefacts).
        newest = max((MODEL_DIR / fn).stat().st_mtime for fn in TASKS.values())
        trained = pd.Timestamp(newest, unit="s", tz="UTC").tz_convert("Asia/Dhaka")
        ds = artifact("dataset.json")
        return {
            "app_version": APP_VERSION,
            "models": [{"task": t, "model": MODEL_NAMES[t], "file": fn,
                        "size_kb": round((MODEL_DIR / fn).stat().st_size / 1024)}
                       for t, fn in TASKS.items()],
            "libraries": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
            "dataset": {"file": "global_climate_health_impact_tracker_2015_2025.csv",
                        "rows": ds["rows"], "countries": ds["countries"],
                        "start": ds["start"], "end": ds["end"]},
            "last_training": trained.strftime("%Y-%m-%d %H:%M"),
            "disclaimer": advisory.DISCLAIMER,
        }

