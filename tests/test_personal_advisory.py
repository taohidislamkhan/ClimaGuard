"""Personal advisory engine: matrix, scenarios, privacy, and text safety.

Scenario tests call the pure engine (``personal_rules.evaluate``) with fixed
inputs; API tests use the offline Open-Meteo fakes from ``conftest.py``.
"""

import logging
import re
import sqlite3
from datetime import datetime

import pytest

from dashboard import personal_rules as pr
from dashboard import weather_client
from dashboard.storage import DEFAULT_PROFILE
from dashboard.weather_client import fetch_forecast as real_fetch_forecast

NOW = datetime(2026, 10, 2, 8, 0)
HEAT_THR = 35.0


def profile(**kw) -> dict:
    return {**DEFAULT_PROFILE, "age": 30, **kw}


def regional(**bands) -> dict:
    return {d: {"band": bands.get(d, "Low"), "score": {"Low": 20, "Moderate": 50, "High": 80}[bands.get(d, "Low")],
                "driver": "PM2.5"} for d in pr.DISEASES}


def hours(feels=None, pm25=None) -> list[dict]:
    """24 hourly rows from NOW; ``feels`` / ``pm25`` map hour-of-day -> value."""
    out = []
    for i in range(24):
        h = (NOW.hour + i) % 24
        day = NOW.day + (NOW.hour + i) // 24
        out.append({"time": f"2026-10-{day:02d}T{h:02d}:00:00",
                    "temp": 30.0, "feels_like": (feels or {}).get(h, 33.0),
                    "pm25": (pm25 or {}).get(h, 12.0), "aqi": 50.0})
    return out


def run(p, reg=None, pm25=12.0, aqi=40.0, tmax=31.0, hrs=None, lang="en"):
    return pr.evaluate(p, reg or regional(), {"pm25": pm25, "aqi": aqi, "rain_week": 10.0},
                       {"tmax_today": tmax, "hours": hrs if hrs is not None else hours()},
                       HEAT_THR, lang, now=NOW)


def by(out, disease):
    return next(x for x in out["diseases"] if x["disease"] == disease)


# -- matrix ------------------------------------------------------------------
EXPECTED = {("Low", "normal"): "Low", ("Low", "elevated"): "Low", ("Low", "high"): "Moderate",
            ("Moderate", "normal"): "Moderate", ("Moderate", "elevated"): "Moderate", ("Moderate", "high"): "High",
            ("High", "normal"): "High", ("High", "elevated"): "High", ("High", "high"): "Very High"}


@pytest.mark.parametrize("band,sens", list(EXPECTED))
def test_matrix_every_combination(band, sens):
    assert pr.matrix_level(band, sens) == EXPECTED[(band, sens)]
    # end to end through the engine, on vector-borne (no live overrides apply):
    # nets -> normal, no nets -> elevated, pregnancy -> high
    p = {"normal": profile(), "elevated": profile(mosquito_nets=False),
         "high": profile(pregnancy=True)}[sens]
    v = by(run(p, regional(vector=band)), "vector")
    assert v["sensitivity"] == sens
    assert v["personal_level"] == EXPECTED[(band, sens)]


def test_sensitivity_is_max_over_matching_rules():
    s = pr.sensitivity(profile(age=70, smoking=True, commute="walk"))
    assert s["respiratory"]["sensitivity"] == "high"            # older adult beats smoker / walking
    assert {r["id"] for r in s["respiratory"]["rules"]} == {"resp_older", "resp_smoker", "resp_commute"}


def test_every_rule_is_sourced_and_unique():
    ids = [r["id"] for r in pr.RULES]
    assert len(ids) == len(set(ids))
    for r in pr.RULES:
        assert r["source"] in pr.SOURCES and pr.SOURCES[r["source"]]["url"].startswith("https://")
        assert r["sensitivity"] in ("elevated", "high")
        assert r["disease"] in pr.DISEASES


# -- scenarios -----------------------------------------------------------------
def test_scenario_healthy_adult_low_day_gets_only_general_tips():
    out = run(profile())
    assert [i["disease"] for i in out["items"]] == ["general"]
    assert out["items"][0]["personal_level"] == "Low"
    assert 1 <= len(out["items"][0]["actions"]) <= 3
    assert out["red_flags"] == []
    assert out["disclaimer"]


@pytest.mark.parametrize("band", ["Low", "Moderate"])
def test_scenario_asthma_pm25_80_gets_high_respiratory_and_red_flags(band):
    out = run(profile(asthma=True, consent=True), regional(respiratory=band), pm25=80, aqi=165,
              hrs=hours(pm25={h: 80.0 for h in range(8, 13)}))
    first = out["items"][0]
    assert first["disease"] == "respiratory" and first["personal_level"] == "High"
    assert "PM2.5 is 80 µg/m³" in first["why"] and "asthma" in first["why"]
    assert "Regional respiratory risk" in first["why"] and "PM2.5" in first["why"]
    assert first["when"] == "Limit time outdoors 8 AM–1 PM (PM2.5 forecast above 35 µg/m³)"
    assert 2 <= len(first["actions"]) <= 3
    assert any("call 999" in f for f in out["red_flags"])
    assert any(s["label"].startswith("US EPA AirNow") for s in first["sources"])


def test_scenario_older_adult_no_ac_38c_gets_high_heat_with_time_window():
    p = profile(age=70, has_cooling=False)
    out = run(p, regional(heat="Moderate"), tmax=38.0,
              hrs=hours(feels={13: 43.0, 14: 44.0, 15: 44.0, 16: 43.0}))
    heat = next(i for i in out["items"] if i["disease"] == "heat")
    assert heat["personal_level"] == "High"
    assert heat["when"] == "Avoid outdoor activity 1 PM–5 PM (peak heat, up to 44 °C feels-like)"
    assert "38 °C" in heat["why"] and "65 or older" in heat["why"] and "no AC or fan" in heat["why"]
    assert heat["actions"][0].startswith("Spend the hottest hours")      # rule-specific action leads
    assert any("Confusion" in f for f in out["red_flags"])


def test_moderate_floors_are_a_safety_net_for_a_tuned_matrix(monkeypatch):
    # With the default matrix, Low x high is already Moderate, so the
    # "at least Moderate" floors only bite if params.yaml is tuned down.
    monkeypatch.setitem(pr.MATRIX, "Low", {"normal": "Low", "elevated": "Low", "high": "Low"})
    h = by(run(profile(age=70), regional(heat="Low"), tmax=38.0), "heat")
    assert h["matrix_level"] == "Low" and h["personal_level"] == "Moderate" and h["overrides"]
    h = by(run(profile(age=70), regional(heat="Low"), tmax=34.0), "heat")
    assert h["personal_level"] == "Low" and h["overrides"] == []        # below the 35 °C threshold
    c = by(run(profile(cardiovascular_disease=True, consent=True), pm25=40, aqi=112), "cardio")
    assert c["personal_level"] == "Moderate" and c["overrides"]
    c = by(run(profile(cardiovascular_disease=True, consent=True), pm25=20, aqi=60), "cardio")
    assert c["personal_level"] == "Low"


def test_unhealthy_air_lifts_high_sensitivity_to_high():
    r = by(run(profile(asthma=True, consent=True), regional(respiratory="Low"), pm25=80, aqi=165), "respiratory")
    assert r["matrix_level"] == "Moderate" and r["personal_level"] == "High" and r["overrides"]


def test_air_override_needs_high_sensitivity():
    smoker = by(run(profile(smoking=True), pm25=80, aqi=165), "respiratory")
    assert smoker["sensitivity"] == "elevated" and smoker["personal_level"] == "Low"
    assert smoker["overrides"] == []


def test_scenario_child_tube_well_high_waterborne_week_gets_boil_filter_advice():
    out = run(profile(age=8, water_source="tube_well"), regional(waterborne="High"))
    w = next(i for i in out["items"] if i["disease"] == "waterborne")
    assert w["personal_level"] == "Very High"
    assert any("Boil drinking water" in a and "filter" in a for a in w["actions"])
    assert "tube-well" in w["why"]
    assert any("sunken eyes" in f for f in out["red_flags"])


def test_items_sorted_and_capped():
    p = profile(age=70, asthma=True, pregnancy=True, mosquito_nets=False, water_source="tap",
                weakened_immunity=True, cardiovascular_disease=True, consent=True)
    out = run(p, regional(respiratory="High", vector="Moderate", heat="High", waterborne="Moderate"),
              pm25=80, aqi=165, tmax=38)
    levels = [pr.LEVELS.index(i["personal_level"]) for i in out["items"]]
    assert levels == sorted(levels, reverse=True)
    assert len(out["items"]) <= 5


def test_bangla_toggle_uses_fixed_translations():
    out = run(profile(asthma=True, consent=True), pm25=80, aqi=165, lang="bn")
    first = out["items"][0]
    assert first["title"] == pr.TITLES["respiratory"]["bn"]
    assert first["personal_level_text"] == "উচ্চ"
    assert "৮০" in first["why"]                                          # Bangla digits
    assert out["disclaimer"] == pr.DISCLAIMER["bn"]
    for pair in pr.all_text():
        assert pair["en"].strip() and pair["bn"].strip() and pair["en"] != pair["bn"]


# -- safety: no medication names or doses ------------------------------------------
MEDICATION = re.compile(
    r"\b(paracetamol|acetaminophen|ibuprofen|aspirin|salbutamol|albuterol|inhaler|antibiotics?|"
    r"antihistamines?|steroids?|insulin|metformin|napa|ors|zinc|tablets?|pills?|capsules?|doses?|"
    r"\d+\s?(mg|ml))\b|প্যারাসিটামল|সালবিউটামল|ট্যাবলেট|ওষুধ|ডোজ|ইনহেলার", re.IGNORECASE)


def test_no_action_text_names_a_medication():
    texts = [s for pair in pr.all_text() for s in pair.values()]
    p = profile(age=70, asthma=True, pregnancy=True, mosquito_nets=False, water_source="tap",
                cardiovascular_disease=True, diabetes=True, has_cooling=False, consent=True)
    for lang in ("en", "bn"):
        out = run(p, regional(respiratory="High", vector="High", heat="High", waterborne="High"),
                  pm25=80, aqi=165, tmax=38, lang=lang)
        texts += [a for i in out["items"] for a in i["actions"]] + out["red_flags"]
    bad = [t for t in texts if MEDICATION.search(t)]
    assert bad == []


# -- API ------------------------------------------------------------------------------
def test_api_regional_until_a_profile_is_saved(client):
    d = client.get("/api/advisory/personal?loc=Dhaka").get_json()
    assert d["personalized"] is False and "items" in d
    assert client.post("/api/profile", json={"name": "Rahim"}).status_code == 200
    d = client.get("/api/advisory/personal?loc=Dhaka").get_json()
    assert d["personalized"] is True and 1 <= len(d["items"]) <= 5
    for i in d["items"]:
        assert {"disease", "personal_level", "title", "actions", "why", "when", "sources"} <= set(i)


def test_api_asthma_on_polluted_day(client):
    # fake live air: PM2.5 78 µg/m³, AQI 142 -> above EPA "unhealthy" PM2.5 cut (55.4)
    client.post("/api/profile", json={"asthma": True, "consent": True})
    d = client.get("/api/advisory/personal?loc=Dhaka").get_json()
    resp = next(i for i in d["items"] if i["disease"] == "respiratory")
    assert pr.LEVELS.index(resp["personal_level"]) >= pr.LEVELS.index("High")
    assert "PM2.5 is 78 µg/m³" in resp["why"] and d["red_flags"]
    bn = client.get("/api/advisory/personal?loc=Dhaka&lang=bn").get_json()
    assert bn["lang"] == "bn" and bn["disclaimer"] == pr.DISCLAIMER["bn"]


def test_api_my_risk_and_preview_and_methodology(client):
    client.post("/api/profile", json={"age": 70, "has_cooling": False})
    dz = client.get("/api/disease/heat?loc=Dhaka").get_json()["personal"]
    assert dz["personalized"] and {r["id"] for r in dz["rules_fired"]} == {"heat_older", "heat_no_cooling"}
    pv = client.post("/api/profile/preview?loc=Dhaka", json={"age": 30, "has_cooling": True}).get_json()
    assert by(pv["advisory"], "heat")["rules_fired"] == []              # preview reflects the unsaved form
    m = client.get("/api/methodology").get_json()["personal_engine"]
    assert len(m["rules"]) == len(pr.RULES) and m["matrix"]["High"]["high"] == "Very High"


def test_condition_needs_consent(client):
    r = client.post("/api/profile", json={"asthma": True})
    assert r.status_code == 400 and "consent" in r.get_json()["detail"]
    assert client.post("/api/profile", json={"asthma": True, "consent": True}).status_code == 200


# -- privacy --------------------------------------------------------------------------
def test_delete_endpoint_wipes_the_row(client, svc, tmp_path):
    client.post("/api/profile", json={"name": "Karim", "diabetes": True, "consent": True})
    assert svc.app.has_profile()
    r = client.delete("/api/profile")
    assert r.status_code == 200 and r.get_json()["deleted"] is True
    assert not svc.app.has_profile()
    assert sqlite3.connect(tmp_path / "h.db").execute("SELECT COUNT(*) FROM profile").fetchone()[0] == 0
    assert client.get("/api/advisory/personal").get_json()["personalized"] is False
    assert client.get("/api/profile").get_json()["saved"] is False


def test_no_profile_data_in_logs(client, caplog):
    caplog.set_level(logging.DEBUG)
    body = {"name": "Zubaida", "asthma": True, "pregnancy": True, "water_source": "tube_well",
            "consent": True}
    client.post("/api/profile", json=body)
    client.get("/api/advisory/personal?loc=Dhaka")
    client.post("/api/profile/preview?loc=Dhaka", json=body)
    client.get("/api/disease/respiratory?loc=Dhaka")
    client.delete("/api/profile")
    log = caplog.text.lower()
    for secret in ("zubaida", "asthma", "pregnan", "tube_well"):
        assert secret not in log


def test_profile_never_sent_to_external_apis(svc, monkeypatch):
    sent = []

    def fake_get_json(url, params):
        sent.append((url, dict(params)))
        if "air-quality" in url:
            t = [f"2026-10-02T{h:02d}:00" for h in range(24)] + [f"2026-10-03T{h:02d}:00" for h in range(24)]
            return {"hourly": {"time": t, "pm2_5": [40.0] * 48, "us_aqi": [110.0] * 48}}
        t = [f"2026-10-02T{h:02d}:00" for h in range(24)] + [f"2026-10-03T{h:02d}:00" for h in range(24)]
        return {"daily": {"time": ["2026-10-02"], "temperature_2m_max": [36.0], "temperature_2m_min": [27.0],
                          "precipitation_sum": [0.0], "precipitation_probability_max": [10]},
                "hourly": {"time": t, "temperature_2m": [33.0] * 48, "apparent_temperature": [39.0] * 48}}

    monkeypatch.setattr(weather_client, "_get_json", fake_get_json)
    monkeypatch.setattr(weather_client, "fetch_forecast", real_fetch_forecast)
    svc.app.save_profile({"name": "Nasrin", "asthma": True, "pregnancy": True, "consent": True,
                          "water_source": "tap"})
    out = svc.personal_advisory("Dhaka")
    assert out["personalized"] and sent
    for url, params in sent:
        assert set(params) <= {"latitude", "longitude", "timezone", "forecast_days", "daily", "hourly"}
        assert not re.search(r"nasrin|asthma|pregnan|tap\b", str(params), re.IGNORECASE)


def test_all_day_pollution_window_is_not_a_zero_length_range():
    out = run(profile(asthma=True, consent=True), pm25=80, aqi=165,
              hrs=hours(pm25={h: 70.0 for h in range(24)}))
    assert by(out, "respiratory")["when"].startswith("PM2.5 is forecast above 35 µg/m³ for the next 24 hours")
