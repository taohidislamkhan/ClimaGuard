import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def store():
    from dashboard.inference import ModelStore
    return ModelStore()


from dashboard import service as service_mod  # noqa: E402
from dashboard import weather_client  # noqa: E402
from dashboard.history import History  # noqa: E402
from dashboard.storage import AppStore  # noqa: E402
from dashboard.weather_client import Climatology, DIVISIONS, LiveLocation  # noqa: E402


# -- offline fakes for Open-Meteo (shared by the API and advisory tests) ------
def fake_climatology(store):
    real = np.sort(np.random.default_rng(1).normal(26, 4, 564))
    return Climatology(real, np.sort(store.country["temperature_celsius"].to_numpy()), 35.0)


def fake_live(store, clim):
    """13 dataset weeks (temperature converted back to real °C) per division."""
    hist = store.country_history(13)
    weekly = hist[["temperature_celsius", "pm25_ugm3", "air_quality_index",
                   "precipitation_mm", "heat_wave_days"]].copy()
    weekly["temperature_celsius"] = weekly["temperature_celsius"].map(clim.to_real_temp)
    cur = {"temperature": 31.0, "feels_like": 36.0, "humidity": 78, "wind_speed": 12, "rainfall": 12,
           "uv_index": 6, "pm25": 78, "pm10": 120, "aqi": 142, "time": "2026-09-25T23:45"}
    prev = {**cur, "temperature": 30.0, "pm25": 70, "aqi": 130}

    def fetch(_threshold):
        return {n: LiveLocation(n, *DIVISIONS[n], weekly.copy(), dict(cur), dict(prev))
                for n in DIVISIONS}
    return fetch


def fake_forecast(name, heatwave_tmax):
    days = [{"date": f"2026-09-{27 + i:02d}", "tmax": 33.0 + i, "tmin": 26.0, "rain": 2.0,
             "rain_prob": 40, "heatwave": 33.0 + i > heatwave_tmax} for i in range(7)]
    hours = [{"time": f"2026-09-27T{h % 24:02d}:00:00", "pm25": 20.0 + h, "aqi": 60.0 + h}
             for h in range(72)]
    return {"days": days, "hours": hours, "heatwave_tmax": heatwave_tmax,
            "heatwave_days": sum(d["heatwave"] for d in days)}


@pytest.fixture()
def svc(store, tmp_path, monkeypatch):
    s = service_mod.DashboardService(history=History(tmp_path / "h.db"),
                                     app_store=AppStore(tmp_path / "h.db"), store=store)
    s._clim = fake_climatology(store)
    monkeypatch.setattr(weather_client, "fetch_live", fake_live(store, s._clim))
    monkeypatch.setattr(weather_client, "fetch_forecast", fake_forecast)
    return s


@pytest.fixture()
def client(svc):
    from app import create_app
    return create_app(svc).test_client()
