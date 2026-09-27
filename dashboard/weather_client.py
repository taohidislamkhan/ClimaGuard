"""Open-Meteo client (forecast + air-quality + archive), 30-minute cache.

Multi-location requests fetch all eight divisions in one HTTP call per API.

Temperature calibration
-----------------------
The dataset's ``temperature_celsius`` is not on a real Celsius scale
(Bangladesh averages 11 °C there, the UK -7.9 °C). Live readings are mapped
onto the dataset scale by **quantile mapping**: the live weekly mean's
percentile within Dhaka's real 2015–2025 weekly climatology (Open-Meteo
archive, fetched once and cached on disk) is looked up in the dataset's
Bangladesh temperature distribution. Heat-wave days use the same archive:
a day counts when its max temperature exceeds the 95th percentile of all
2015–2024 daily maxima.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .inference import ROOT, N_WEEKS

CACHE_TTL = 30 * 60
TIMEOUT = 12
CLIMATOLOGY_PATH = ROOT / "data" / "cache" / "dhaka_climatology.json"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# The forecast API only keeps ~2 months of daily history; the
# historical-forecast API covers the full 13-week window up to today.
HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DIVISIONS = {
    "Dhaka":      (23.8103, 90.4125),
    "Chattogram": (22.3569, 91.7832),
    "Rajshahi":   (24.3745, 88.6042),
    "Khulna":     (22.8456, 89.5403),
    "Sylhet":     (24.8949, 91.8687),
    "Barisal":    (22.7010, 90.3535),
    "Rangpur":    (25.7439, 89.2752),
    "Mymensingh": (24.7471, 90.4203),
}

_cache: dict[str, tuple[float, object]] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _get_json(url: str, params: dict) -> object:
    key = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    with _lock:
        _cache[key] = (time.time(), data)
    return data


def _as_list(data: object) -> list[dict]:
    return data if isinstance(data, list) else [data]


# ---------------------------------------------------------------------------
# Climatology (temperature quantile map + heat-wave threshold)
# ---------------------------------------------------------------------------
@dataclass
class Climatology:
    real_weekly_temp: np.ndarray          # sorted, °C
    dataset_weekly_temp: np.ndarray       # sorted, dataset scale
    heatwave_tmax: float                  # °C, daily-max 95th percentile

    def to_model_temp(self, real_c: float) -> float:
        """Real °C -> dataset temperature scale (quantile mapping)."""
        q = np.searchsorted(self.real_weekly_temp, real_c) / len(self.real_weekly_temp)
        return float(np.quantile(self.dataset_weekly_temp, np.clip(q, 0, 1)))

    def to_real_temp(self, model_c: float) -> float:
        """Inverse mapping, used to display dataset weeks in real °C."""
        q = np.searchsorted(self.dataset_weekly_temp, model_c) / len(self.dataset_weekly_temp)
        return float(np.quantile(self.real_weekly_temp, np.clip(q, 0, 1)))


def load_climatology(dataset_temps: np.ndarray) -> Climatology:
    """Dhaka 2015–2025 archive, fetched once and cached under data/cache/."""
    if CLIMATOLOGY_PATH.exists():
        c = json.loads(CLIMATOLOGY_PATH.read_text())
    else:
        lat, lon = DIVISIONS["Dhaka"]
        d = _get_json(ARCHIVE_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": "2015-01-01", "end_date": "2025-10-19",
            "daily": "temperature_2m_mean,temperature_2m_max",
            "timezone": "Asia/Dhaka",
        })["daily"]
        df = pd.DataFrame(d).dropna()
        df["time"] = pd.to_datetime(df["time"])
        weekly = df.set_index("time")["temperature_2m_mean"].resample("W-SUN").mean().dropna()
        tmax = df.loc[df["time"] < "2025-01-01", "temperature_2m_max"]
        c = {"weekly_temp": [round(v, 3) for v in weekly.tolist()],
             "heatwave_tmax": float(np.quantile(tmax, 0.95)),
             "source": "Open-Meteo archive, Dhaka 2015-01-01..2025-10-19"}
        CLIMATOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLIMATOLOGY_PATH.write_text(json.dumps(c))
    return Climatology(np.sort(np.asarray(c["weekly_temp"], float)),
                       np.sort(np.asarray(dataset_temps, float)),
                       float(c["heatwave_tmax"]))


# ---------------------------------------------------------------------------
# Live data
# ---------------------------------------------------------------------------
@dataclass
class LiveLocation:
    name: str
    lat: float
    lon: float
    weekly: pd.DataFrame                  # N_WEEKS rows, real units (oldest first)
    current: dict = field(default_factory=dict)
    previous: dict = field(default_factory=dict)   # same fields, 24 h earlier


def _weekly(daily: pd.DataFrame, hourly_aq: pd.DataFrame, today: pd.Timestamp,
            heatwave_tmax: float) -> pd.DataFrame:
    """Aggregate to 7-day blocks ending today (block 0 = current week)."""
    rows = []
    for k in range(N_WEEKS - 1, -1, -1):
        end = today - pd.Timedelta(days=7 * k)
        start = end - pd.Timedelta(days=6)
        d = daily[(daily["time"] >= start) & (daily["time"] <= end)]
        a = hourly_aq[(hourly_aq["time"] >= start) &
                      (hourly_aq["time"] < end + pd.Timedelta(days=1))]
        rows.append({
            "week_end": end,
            "temperature_celsius": d["temperature_2m_mean"].mean(),
            "precipitation_mm": d["precipitation_sum"].sum(min_count=1),
            "heat_wave_days": int((d["temperature_2m_max"] > heatwave_tmax).sum()),
            "pm25_ugm3": a["pm2_5"].mean(),
            "air_quality_index": a["us_aqi"].mean(),
        })
    return pd.DataFrame(rows).ffill().bfill()


def fetch_live(heatwave_tmax: float) -> dict[str, LiveLocation]:
    """Live weather + air quality for all divisions (three parallel HTTP calls)."""
    names = list(DIVISIONS)
    lats = ",".join(str(DIVISIONS[n][0]) for n in names)
    lons = ",".join(str(DIVISIONS[n][1]) for n in names)
    common = {"latitude": lats, "longitude": lons, "timezone": "Asia/Dhaka"}
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Dhaka")).date())
    requests_ = [
        (FORECAST_URL, {
            **common, "past_days": 2, "forecast_days": 1,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,precipitation",
            "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
            "daily": "precipitation_sum,uv_index_max",
        }),
        (HISTORY_URL, {
            **common,
            "start_date": (today - pd.Timedelta(days=7 * N_WEEKS - 1)).date().isoformat(),
            "end_date": today.date().isoformat(),
            "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum",
        }),
        (AIR_URL, {
            **common, "past_days": 92, "forecast_days": 1,
            "current": "pm2_5,pm10,us_aqi",
            "hourly": "pm2_5,pm10,us_aqi",
        }),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        fc, hist, aq = (_as_list(r) for r in pool.map(lambda a: _get_json(*a), requests_))

    out: dict[str, LiveLocation] = {}
    for name, f, h, a in zip(names, fc, hist, aq):
        daily = pd.DataFrame(f["daily"])
        daily["time"] = pd.to_datetime(daily["time"])
        hdaily = pd.DataFrame(h["daily"])
        hdaily["time"] = pd.to_datetime(hdaily["time"])
        hourly = pd.DataFrame(f["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"])
        haq = pd.DataFrame(a["hourly"])
        haq["time"] = pd.to_datetime(haq["time"])

        now = pd.Timestamp(f["current"]["time"])
        weekly = _weekly(hdaily, haq, today, heatwave_tmax)

        def at(df: pd.DataFrame, t: pd.Timestamp, col: str) -> float | None:
            sub = df[df["time"] <= t].dropna(subset=[col])
            return float(sub[col].iloc[-1]) if len(sub) else None

        prev_t = now - pd.Timedelta(hours=24)
        day = daily[daily["time"] == today]
        yday = daily[daily["time"] == today - pd.Timedelta(days=1)]
        current = {
            "temperature": f["current"]["temperature_2m"],
            "feels_like": f["current"].get("apparent_temperature"),
            "humidity": f["current"]["relative_humidity_2m"],
            "wind_speed": f["current"]["wind_speed_10m"],
            "rainfall": float(day["precipitation_sum"].iloc[0]) if len(day) else None,
            "uv_index": float(day["uv_index_max"].iloc[0]) if len(day) else None,
            "pm25": a["current"]["pm2_5"],
            "pm10": a["current"].get("pm10"),
            "aqi": a["current"]["us_aqi"],
            "time": now.isoformat(),
        }
        previous = {
            "temperature": at(hourly, prev_t, "temperature_2m"),
            "feels_like": at(hourly, prev_t, "apparent_temperature"),
            "humidity": at(hourly, prev_t, "relative_humidity_2m"),
            "wind_speed": at(hourly, prev_t, "wind_speed_10m"),
            "rainfall": float(yday["precipitation_sum"].iloc[0]) if len(yday) else None,
            "uv_index": float(yday["uv_index_max"].iloc[0]) if len(yday) else None,
            "pm25": at(haq, prev_t, "pm2_5"),
            "pm10": at(haq, prev_t, "pm10"),
            "aqi": at(haq, prev_t, "us_aqi"),
        }
        out[name] = LiveLocation(name, *DIVISIONS[name], weekly, current, previous)
    return out


def fetch_forecast(name: str, heatwave_tmax: float) -> dict:
    """7-day daily weather + next-72-hour air quality for one division."""
    lat, lon = DIVISIONS[name]
    common = {"latitude": lat, "longitude": lon, "timezone": "Asia/Dhaka"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        fc, aq = pool.map(lambda a: _get_json(*a), [
            (FORECAST_URL, {**common, "forecast_days": 7,
                            "daily": "temperature_2m_max,temperature_2m_min,"
                                     "precipitation_sum,precipitation_probability_max"}),
            (AIR_URL, {**common, "forecast_days": 4, "hourly": "pm2_5,us_aqi"}),
        ])
    d = fc["daily"]
    days = [{"date": t, "tmax": hi, "tmin": lo, "rain": r, "rain_prob": pp,
             "heatwave": hi is not None and hi > heatwave_tmax}
            for t, hi, lo, r, pp in zip(d["time"], d["temperature_2m_max"], d["temperature_2m_min"],
                                        d["precipitation_sum"], d["precipitation_probability_max"])]
    now = pd.Timestamp(datetime.now(ZoneInfo("Asia/Dhaka")).replace(tzinfo=None)).floor("h")
    h = pd.DataFrame(aq["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h = h[(h["time"] >= now) & (h["time"] < now + pd.Timedelta(hours=72))]
    hours = [{"time": t.isoformat(), "pm25": None if pd.isna(p) else float(p),
              "aqi": None if pd.isna(q) else float(q)}
             for t, p, q in zip(h["time"], h["pm2_5"], h["us_aqi"])]
    return {"days": days, "hours": hours, "heatwave_tmax": heatwave_tmax,
            "heatwave_days": sum(x["heatwave"] for x in days)}
