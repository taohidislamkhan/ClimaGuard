"""Assemble the single JSON payload behind ``GET /api/dashboard``.

One *model run* covers all eight divisions at once (a batch of eight
feature rows through each model) and is cached for 30 minutes; the
requested location, the signed-in user's profile (passed in by the caller,
None for guests) and the history lookups are applied per request on top of it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import advisory, personal_rules, scoring, shap_utils, weather_client
from .history import History
from .inference import MODEL_NAMES, TARGETS, ModelStore, Prediction
from .storage import DEFAULT_PROFILE, DEFAULT_SETTINGS

TZ = ZoneInfo("Asia/Dhaka")
DISEASE_ORDER = ["respiratory", "vector", "heat", "waterborne", "cardio"]
DISEASE_LABELS = {"respiratory": "Respiratory", "vector": "Vector-borne",
                  "heat": "Heat-related", "waterborne": "Waterborne",
                  "cardio": "Cardiovascular"}
ENV_LABELS = {"pm25": "PM2.5", "aqi": "Air Quality Index",
              "temperature": "Temperature", "rainfall": "Rainfall"}

def aqi_category(aqi: float | None) -> dict:
    """US EPA AQI categories mapped onto the three risk colours
    (Moderate and Unhealthy-for-sensitive-groups share the amber band)."""
    if aqi is None:
        return {"label": "Unknown", "level": None}
    for hi, label, level in [(50, "Good", "Low"), (100, "Moderate", "Moderate"),
                             (150, "Unhealthy for sensitive groups", "Moderate"),
                             (200, "Unhealthy", "High"), (300, "Very unhealthy", "High")]:
        if aqi <= hi:
            return {"label": label, "level": level}
    return {"label": "Hazardous", "level": "High"}


@dataclass
class ModelRun:
    live: bool
    computed_at: datetime
    date: str                                # snapshot date (Asia/Dhaka)
    rows: dict[str, pd.DataFrame]            # location -> 1-row feature matrix
    preds: dict[str, Prediction]
    env_now: dict[str, dict]                 # location -> current display values
    env_prev: dict[str, dict]
    env_week: dict[str, dict]                # location -> week-level model inputs (real units)


class DashboardService:
    def __init__(self, history: History | None = None,
                 store: ModelStore | None = None) -> None:
        self.store = store or ModelStore()
        self.history = history or History()
        self.ttl_minutes = DEFAULT_SETTINGS["refresh_minutes"]   # shared by all users
        self._clim = None
        self._run: ModelRun | None = None
        self._lock = threading.Lock()
        self._drivers: dict[tuple, str | None] = {}

    @property
    def run_ttl(self) -> int:
        """Seconds a model run stays fresh. One run serves every user, so this
        is server-wide; a user's refresh interval drives their page reloads."""
        return int(self.ttl_minutes) * 60

    def run_status(self) -> dict | None:
        """Live/fallback state and age of the cached model run (admin status card)."""
        run = self._run
        if run is None:
            return None
        return {"live": run.live, "computed_at": run.computed_at.isoformat(timespec="seconds"),
                "age_seconds": round(time.time() - run.computed_at.timestamp())}

    def clear_cache(self) -> None:
        """Drop cached weather and the cached model run; the next request refetches."""
        weather_client.clear_cache()
        with self._lock:
            self._run = None

    # -- climatology / live data ---------------------------------------------
    def climatology(self):
        if self._clim is None:
            self._clim = weather_client.load_climatology(
                self.store.country["temperature_celsius"].to_numpy())
        return self._clim

    def _live_run(self) -> ModelRun:
        clim = self.climatology()
        live = weather_client.fetch_live(clim.heatwave_tmax)
        now = datetime.now(TZ)
        names = list(live)
        rows = []
        env_week = {}
        for name in names:
            weekly = live[name].weekly.copy()
            real_temp = weekly["temperature_celsius"].copy()
            weekly["temperature_celsius"] = real_temp.map(clim.to_model_temp)
            rows.append(self.store.build_row(weekly, pd.Timestamp(now.date())))
            last = live[name].weekly.iloc[-1]
            env_week[name] = {"pm25": float(last["pm25_ugm3"]),
                              "aqi": float(last["air_quality_index"]),
                              "temperature": float(real_temp.iloc[-1]),
                              "rainfall": float(last["precipitation_mm"])}
        X = self.store.to_matrix(rows)
        preds = self.store.predict(X)
        return ModelRun(
            live=True, computed_at=now, date=now.date().isoformat(),
            rows={n: X.iloc[[i]] for i, n in enumerate(names)},
            preds=dict(zip(names, preds)),
            env_now={n: live[n].current for n in names},
            env_prev={n: live[n].previous for n in names},
            env_week=env_week,
        )

    def _fallback_run(self) -> ModelRun:
        """Open-Meteo unreachable: latest dataset week for every division."""
        t = self.store.template
        X = self.store.to_matrix([t])
        pred = self.store.predict(X)[0]
        real_t = self._real_temp(float(t["temperature_celsius"]))
        env = {"pm25": float(t["pm25_ugm3"]), "aqi": float(t["air_quality_index"]),
               "temperature": real_t, "rainfall": float(t["precipitation_mm"])}
        now_env = {**env, "humidity": None, "wind_speed": None, "uv_index": None,
                   "time": pd.Timestamp(t["date"]).isoformat()}
        names = list(weather_client.DIVISIONS)
        return ModelRun(
            live=False, computed_at=datetime.now(TZ),
            date=pd.Timestamp(t["date"]).date().isoformat(),
            rows={n: X for n in names}, preds={n: pred for n in names},
            env_now={n: now_env for n in names}, env_prev={n: {} for n in names},
            env_week={n: env for n in names},
        )

    def _real_temp(self, model_temp: float) -> float | None:
        try:
            return self.climatology().to_real_temp(model_temp)
        except Exception:
            return None

    def _refresh(self, force: bool = False) -> ModelRun:
        if force:
            weather_client.clear_cache()
        try:
            run = self._live_run()
        except Exception:
            run = self._fallback_run()
        self._run = run
        return run

    def model_run(self, force: bool = False) -> ModelRun:
        """Cached model run. A stale run is served immediately while a
        background thread refreshes it (stale-while-revalidate), so page
        loads never block on Open-Meteo after the first run."""
        if force or self._run is None:
            with self._lock:
                if force or self._run is None:
                    return self._refresh(force)
                return self._run
        stale = time.time() - self._run.computed_at.timestamp() >= self.run_ttl
        if stale and self._lock.acquire(blocking=False):
            def worker():
                try:
                    self._refresh(force=True)
                finally:
                    self._lock.release()
            threading.Thread(target=worker, daemon=True).start()
        return self._run

    # -- scores ----------------------------------------------------------------
    def model_scores(self, pred: Prediction) -> dict[str, float]:
        s = {k: self.store.percentile(k, v) for k, v in pred.diseases.items()}
        s["overall"] = scoring.overall_score(pred.proba)
        return s

    def ensure_backfill(self, location: str) -> None:
        if self.history.has_rows(location):
            return
        hist = self.store.country_history(7)
        preds = self.store.predict(self.store.to_matrix(hist))
        for (_, row), pred in zip(hist.iterrows(), preds):
            self.history.upsert(
                location, pd.Timestamp(row["date"]).date().isoformat(), "backfill",
                scoring.overall_level(pred.proba), self.model_scores(pred),
                {"pm25": float(row["pm25_ugm3"]), "aqi": float(row["air_quality_index"]),
                 "temperature": self._real_temp(float(row["temperature_celsius"])),
                 "rainfall": float(row["precipitation_mm"])})

    # -- payload ---------------------------------------------------------------
    def dashboard(self, location: str = "Dhaka", force: bool = False,
                  profile: dict | None = None) -> dict:
        """Dashboard payload. ``profile`` is the signed-in user's saved profile;
        None (guest, or no profile yet) gives the regional view without
        personal score adjustments."""
        if location not in weather_client.DIVISIONS:
            raise KeyError(location)
        run = self.model_run(force)
        pred = run.preds[location]
        scores = self.model_scores(pred)
        level = scoring.overall_level(pred.proba)

        self.ensure_backfill(location)
        if run.live:
            self.history.upsert(location, run.date, "live", level, scores,
                                run.env_week[location])
        prev = self.history.previous(location, run.date)
        recent = self.history.recent(location, 7)
        demo_history = (not run.live) or any(r["source"] == "backfill" for r in recent)

        adjustments = scoring.personal_adjustments(profile) if profile else {}

        def change(key: str) -> float | None:
            """Change in score points versus the previous snapshot."""
            return _r(scores[key] - prev[key]) if prev else None

        compare = _compare_label(prev, run.date)

        overall = {
            "score": round(scores["overall"]),
            "level": level,
            "probabilities": {("Moderate" if k == "Medium" else k): round(v, 3)
                              for k, v in pred.proba.items()},
            "change_points": change("overall"),
            "compare_label": compare,
            "model": MODEL_NAMES["overall"],
            "formula": "100 × (0.5·P(Medium) + 1.0·P(High))",
            "demo_change": demo_history,
        }

        diseases = []
        for key in DISEASE_ORDER:
            adj = adjustments.get(key, {"points": 0, "reasons": []})
            score = scoring.apply_adjustment(scores[key], adj["points"])
            diseases.append({
                "key": key, "label": DISEASE_LABELS[key],
                "score": round(score), "model_score": round(scores[key]),
                "level": scoring.band(score),
                "adjustment": adj,
                "change_points": change(key),
                "compare_label": compare,
                "sparkline": [round(r[key]) for r in recent],
                "model": MODEL_NAMES[key], "target": TARGETS[key],
                "prediction": round(pred.diseases[key], 2),
                "note": _disease_note(key),
            })

        factors = shap_utils.top_factors(self.store.models["overall"], run.rows[location])

        now_env, prev_env = run.env_now[location], run.env_prev[location]
        aqi_cat = aqi_category(now_env.get("aqi"))
        environment = {
            "location": location,
            "live": run.live,
            "observed_at": now_env.get("time"),
            "values": {k: now_env.get(k) for k in
                       ["temperature", "humidity", "aqi", "pm25", "rainfall", "wind_speed", "uv_index"]},
            "trend": {k: _trend(now_env.get(k), prev_env.get(k)) for k in
                      ["temperature", "humidity", "aqi", "pm25", "rainfall", "wind_speed", "uv_index"]},
            "aqi_category": aqi_cat,
            "model_inputs": ["temperature", "aqi", "pm25", "rainfall"],
            "display_only": ["humidity", "wind_speed", "uv_index"],
        }

        trend = {"labels": [_label(r) for r in recent],
                 "sources": [r["source"] for r in recent],
                 "demo": demo_history}
        for key in ["overall", "respiratory", "heat", "vector", "waterborne"]:
            trend[key] = [round(r[key]) for r in recent]

        changes = {
            "points": overall["change_points"],
            "compare_label": compare,
            "previous_date": prev["date"] if prev else None,
            "previous_source": prev["source"] if prev else None,
            "demo": demo_history,
            "contributors": sorted(
                [{"key": k, "label": ENV_LABELS[k],
                  "change_pct": _r(scoring.pct_change(run.env_week[location].get(k),
                                                      prev.get(k) if prev else None))}
                 for k in ENV_LABELS],
                key=lambda c: -abs(c["change_pct"] or 0))[:3],
        }

        risk_map = []
        for name, (lat, lon) in weather_client.DIVISIONS.items():
            p = run.preds[name]
            risk_map.append({"name": name, "lat": lat, "lon": lon,
                             "score": round(scoring.overall_score(p.proba)),
                             "level": scoring.overall_level(p.proba)})

        return {
            "location": {"name": location, "country": "Bangladesh"},
            "live": run.live,
            "data_date": run.date,
            "overall": overall,
            "diseases": diseases,
            "shap_factors": factors,
            "environment": environment,
            "trend": trend,
            "advisories": advisory.build_advisories(
                "Medium" if level == "Moderate" else level, pred.proba,
                scores, now_env.get("pm25")),
            "disclaimer": advisory.DISCLAIMER,
            "advisory_footer": advisory.FOOTER,
            "changes": changes,
            "map": risk_map,
            "profile": profile,
            "updated_at": run.computed_at.isoformat(timespec="seconds"),
        }

    # -- personal advisory (rule-based; the profile never enters a model) ------
    def _top_driver(self, run: ModelRun, loc: str, key: str) -> str | None:
        """Friendly label of the feature group that raises this disease's prediction most."""
        ck = (run.computed_at, loc, key)
        if ck not in self._drivers:
            rows = shap_utils.grouped_signed(self.store.models[key], run.rows[loc], k=8)
            up = [r for r in rows if r["shap"] > 0]
            self._drivers[ck] = up[0]["label"] if up else None
        return self._drivers[ck]

    def personal_advisory(self, location: str, profile: dict | None = None,
                          lang: str = "en") -> dict:
        """Personal advisory for ``profile`` (the caller's saved or previewed one).

        ``personalized`` is False when no profile is given; the dashboard then
        keeps the regional advisories."""
        if location not in weather_client.DIVISIONS:
            raise KeyError(location)
        personalized = profile is not None
        profile = profile if profile is not None else DEFAULT_PROFILE
        run = self.model_run()
        scores = self.model_scores(run.preds[location])
        now_env = run.env_now[location]
        regional = {k: {"band": scoring.band(scores[k]), "score": round(scores[k]),
                        "driver": self._top_driver(run, location, k)}
                    for k in ("respiratory", "vector", "heat", "waterborne")}
        # The cardio model has no skill (R² ~ 0): its regional band is the live
        # US EPA AQI category instead (PM2.5 is the EPA's cardiac pathway).
        regional["cardio"] = {"band": aqi_category(now_env.get("aqi"))["level"] or scoring.LOW,
                              "score": round(scores["cardio"]), "driver": None}
        heat_thr = self.climatology().heatwave_tmax
        forecast = None
        if run.live:
            try:
                f = weather_client.fetch_forecast(location, heat_thr)
                forecast = {"tmax_today": f["days"][0]["tmax"] if f["days"] else None,
                            "hours": f["hours"]}
            except Exception:       # forecast is optional: no "when" window, no heat override
                forecast = None
        live = {"pm25": now_env.get("pm25"), "aqi": now_env.get("aqi"),
                "rain_week": run.env_week[location].get("rainfall")}
        out = personal_rules.evaluate(profile, regional, live, forecast, heat_thr, lang,
                                      now=datetime.now(TZ).replace(tzinfo=None))
        return {"personalized": personalized, "location": location, "live": run.live,
                "updated_at": run.computed_at.isoformat(timespec="seconds"),
                "heat_threshold": heat_thr, **out}


def _r(x: float | None, nd: int = 1) -> float | None:
    return None if x is None else round(float(x), nd)


def _compare_label(prev: dict | None, today: str) -> str:
    """Honest name for what a change is measured against."""
    if not prev:
        return "no earlier snapshot"
    if prev["source"] == "backfill":
        return "vs last week (dataset)"
    if pd.Timestamp(today) - pd.Timestamp(prev["date"]) == pd.Timedelta(days=1):
        return "vs yesterday"
    return "vs last snapshot"


def _trend(now: float | None, prev: float | None) -> str | None:
    if now is None or prev is None:
        return None
    if abs(now - prev) < 1e-9:
        return "flat"
    return "up" if now > prev else "down"


def _label(row: dict) -> str:
    d = pd.Timestamp(row["date"])
    return d.strftime("%a") if row["source"] == "live" else d.strftime("%b %d")


def _disease_note(key: str) -> str | None:
    if key == "cardio":
        return "Cardio model (ElasticNet) has ~0 R² on validation; its score barely moves."
    if key == "heat":
        return "About half of training weeks have 0 heat admissions, so scores cluster near 0 or 50+."
    return None
