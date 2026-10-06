"""API tests. Open-Meteo is replaced by fakes so tests run offline."""

import time

import pytest

from dashboard import weather_client

TOP_KEYS = {"overall", "diseases", "shap_factors", "environment", "trend",
            "advisories", "changes", "map", "updated_at"}


def test_dashboard_schema(client):
    r = client.get("/api/dashboard?location=Dhaka")
    assert r.status_code == 200
    d = r.get_json()
    assert TOP_KEYS <= set(d)
    assert d["live"] is True
    assert 0 <= d["overall"]["score"] <= 100
    assert d["overall"]["level"] in {"Low", "Moderate", "High"}
    assert [x["key"] for x in d["diseases"]] == ["respiratory", "vector", "heat", "waterborne", "cardio"]
    for x in d["diseases"]:
        assert 0 <= x["score"] <= 100
        assert x["level"] in {"Low", "Moderate", "High"}
    assert len(d["shap_factors"]) == 6
    assert set(d["trend"]) >= {"overall", "respiratory", "heat", "vector", "waterborne"}
    assert len(d["map"]) == 8
    assert d["disclaimer"] == "This is an environmental risk estimate, not a medical diagnosis."


def test_overall_score_matches_probabilities(client):
    d = client.get("/api/dashboard").get_json()
    p = d["overall"]["probabilities"]
    assert d["overall"]["score"] == pytest.approx(100 * (0.5 * p["Moderate"] + p["High"]), abs=1)


def test_backfill_then_demo_flag(client):
    d = client.get("/api/dashboard").get_json()
    assert d["trend"]["demo"] is True
    assert d["trend"]["sources"].count("backfill") == 6
    assert d["trend"]["sources"][-1] == "live"
    assert d["changes"]["previous_source"] == "backfill"


def test_profile_adjusts_scores_not_model(client):
    base = client.get("/api/dashboard").get_json()
    client.post("/api/profile", json={"asthma": True, "age": 70, "outdoor_exposure": "Low", "consent": True})
    after = client.get("/api/dashboard").get_json()
    assert after["overall"] == base["overall"]                    # model output untouched
    resp = {x["key"]: x for x in after["diseases"]}
    assert resp["respiratory"]["adjustment"]["points"] == 5
    assert resp["cardio"]["adjustment"]["points"] == 5
    assert resp["respiratory"]["model_score"] == {x["key"]: x for x in base["diseases"]}["respiratory"]["model_score"]


def test_profile_rejects_unknown_location(client):
    assert client.post("/api/profile", json={"location": "Paris"}).status_code == 400
    assert client.get("/api/dashboard?location=Paris").status_code == 400


def test_recalculate_same_schema(client):
    d = client.post("/api/recalculate", json={"location": "Sylhet"}).get_json()
    assert TOP_KEYS <= set(d)
    assert d["location"]["name"] == "Sylhet"


def test_fallback_when_open_meteo_down(client, monkeypatch):
    def boom(_):
        raise ConnectionError("offline")
    monkeypatch.setattr(weather_client, "fetch_live", boom)
    d = client.post("/api/recalculate").get_json()
    assert d["live"] is False
    assert d["data_date"] == "2025-10-19"
    assert TOP_KEYS <= set(d)


def test_inference_latency(client):
    client.get("/api/dashboard")                                   # warm-up
    t = time.perf_counter()
    client.post("/api/recalculate")                                 # fakes: pure inference
    assert time.perf_counter() - t < 0.8


def test_pages_render(client):
    for path in ["/", "/risk-map", "/risk-history", "/how-it-works", "/my-risk", "/settings"]:
        assert client.get(path).status_code == 200, path


def test_changes_are_points_with_honest_label(client):
    d = client.get("/api/dashboard").get_json()
    assert d["overall"]["compare_label"] == "vs last week (dataset)"
    for x in d["diseases"]:
        assert "change_pct" not in x
        assert x["compare_label"] == "vs last week (dataset)"
        assert x["change_points"] is None or -100 <= x["change_points"] <= 100


def test_autoregressive_factors_flagged(store):
    from dashboard import shap_utils
    x = store.to_matrix([store.template])
    factors = shap_utils.top_factors(store.models["overall"], x, k=50)
    flagged = {f["label"] for f in factors if f["autoregressive"]}
    assert flagged <= shap_utils.AUTOREGRESSIVE
    assert "Recent waterborne cases" in flagged


def test_methodology_bundle(client):
    m = client.get("/api/methodology").get_json()
    assert m["dataset"]["rows"] == 14100 and m["dataset"]["countries"] == 25
    assert m["cleaning"]["negative_aqi"] > 0 and m["cleaning"]["negative_aqi_after"] == 0
    assert len(m["correlation"]["matrix"]) == len(m["correlation"]["labels"])
    assert m["metrics"]["winners"]["overall_classifier"] in {"logreg", "rf", "xgb", "dt"}
    regs = {r["key"]: r for r in m["regressors"]}
    assert regs["cardio"]["r2"] < 0.4 and regs["cardio"]["reliability"]["level"] == "High"
    assert set(m["shap"]) == {"overall", "respiratory", "vector", "heat", "waterborne", "cardio"}
    assert any(r["disease"] == "Cardiovascular" for r in m["advisory_table"])


def test_disease_deep_dive(client):
    client.post("/api/profile", json={"asthma": True, "outdoor_exposure": "High", "smoking": True, "consent": True})
    d = client.get("/api/disease/respiratory?loc=Dhaka").get_json()
    assert d["adjustment"]["points"] == 10 and d["adjustment"]["capped"]
    assert d["score"] == min(100, d["model_score"] + 10)
    assert 1 <= len(d["shap"]) <= 8
    assert all(set(r) >= {"label", "shap", "autoregressive"} for r in d["shap"])
    assert d["reliability"]["r2"] > 0.4
    assert d["advisory"]["rule"] == "respiratory_high"
    assert d["disclaimer"].startswith("This is an environmental risk estimate")


def test_cardio_low_reliability(client):
    d = client.get("/api/disease/cardio").get_json()
    assert d["reliability"]["r2"] < 0.4
    assert d["reliability"]["label"].startswith("Low reliability")
    assert d["advisory"]["rule"] is None


def test_unknown_disease_404(client):
    r = client.get("/api/disease/flu")
    assert r.status_code == 404 and r.get_json()["error"]


def test_map_division_layers(client):
    for layer in ["overall", "respiratory", "cardio"]:
        d = client.get(f"/api/map?view=division&layer={layer}").get_json()
        assert len(d["items"]) == 8
        assert all(i["level"] in {"Low", "Moderate", "High"} for i in d["items"])


def test_map_country_week_slider(client):
    d = client.get("/api/map?view=country&layer=heat").get_json()
    assert len(d["items"]) == 25 and d["week_index"] == len(d["weeks"]) - 1
    first = client.get("/api/map?view=country&layer=heat&week=0").get_json()
    assert first["week"] == d["weeks"][0]


def test_map_detail(client):
    d = client.get("/api/map/detail?view=division&id=Sylhet").get_json()
    assert len(d["diseases"]) == 5 and len(d["drivers"]) == 3 and d["advisories"]
    c = client.get("/api/map/detail?view=country&id=BGD&week=10").get_json()
    assert c["name"] == "Bangladesh" and len(c["drivers"]) == 3
    assert client.get("/api/map/detail?view=division&id=Paris").status_code == 404


def test_history_ranges(client):
    lens = {}
    for rng in ["4w", "12w", "1y", "all"]:
        h = client.get(f"/api/history?range={rng}").get_json()
        lens[rng] = len(h["dataset"]["dates"])
        assert all(len(h["dataset"][k]) == lens[rng] for k in ["overall", "cardio", "split"])
    assert lens["4w"] == 4 and lens["12w"] == 12 and lens["1y"] == 52 and lens["all"] > 500
    assert client.get("/api/history?range=5y").status_code == 404


def test_snapshots_csv(client):
    client.get("/api/dashboard")
    r = client.get("/api/history.csv?loc=Dhaka")
    lines = r.get_data(as_text=True).strip().splitlines()
    assert r.mimetype == "text/csv" and lines[0].startswith("location,date,source")
    assert 2 <= len(lines) <= 13


def test_test_predictions_and_seasonality(client):
    t = client.get("/api/test-predictions?target=vector").get_json()
    assert t["r2"] > 0.8 and len(t["scatter"]["actual"]) == len(t["scatter"]["pred"])
    assert t["start"] >= "2024-01-01"
    s = client.get("/api/seasonality?scope=bgd").get_json()
    assert len(s["rows"]) == 5 and all(len(r["values"]) == 12 for r in s["rows"])


def test_environment_tiles(client):
    e = client.get("/api/environment?loc=Khulna").get_json()
    tiles = {t["key"]: t for t in e["tiles"]}
    used = {k for k, t in tiles.items() if t["used_by_model"]}
    assert used == {"temperature", "rainfall", "pm25", "aqi", "heat_wave_days"}
    assert {"humidity", "wind_speed", "uv_index", "pm10", "feels_like"} <= set(tiles) - used
    assert tiles["pm25"]["level"] == "High" and "EPA" in tiles["pm25"]["meaning"]   # 78 µg/m³
    assert tiles["temperature"]["model_value"] is not None


def test_forecast_and_failure(client, monkeypatch):
    f = client.get("/api/forecast").get_json()
    assert len(f["days"]) == 7 and len(f["hours"]) == 72
    assert f["thresholds"]["who_pm25_24h"] == 15
    def down(*_):
        raise ConnectionError("offline")
    monkeypatch.setattr(weather_client, "fetch_forecast", down)
    r = client.get("/api/forecast")
    assert r.status_code == 502 and "unavailable" in r.get_json()["detail"]


def test_profile_saved_in_sqlite_and_preview(client, db_profile):
    p = client.post("/api/profile", json={"height_cm": 170, "weight_kg": 95, "diabetes": True, "consent": True}).get_json()
    assert p["bmi"] == round(95 / 1.7 ** 2, 1) and p["bmi"] >= 30
    assert db_profile()["diabetes"] is True                  # persisted, not just in memory
    pv = client.post("/api/profile/preview", json={"age": 70, "diabetes": False}).get_json()
    fired = {r["label"] for r in pv["rules"] if r["fires"]}
    assert "Age 65 or over" in fired and "Diabetes" not in fired and "BMI 30 or over" in fired
    heat = {d["key"]: d for d in pv["diseases"]}["heat"]
    assert heat["points"] == 10 and heat["capped"]            # 3 (outdoor) + 5 (age) + 3 (BMI) = 11 -> 10
    assert db_profile()["age"] != 70                          # preview does not save


def test_profile_validation(client):
    r = client.post("/api/profile", json={"age": 500})
    assert r.status_code == 400 and "Age" in r.get_json()["detail"]


def test_settings_persist_and_apply(client, svc):
    s = client.post("/api/settings", json={"units": "F", "theme": "dark", "refresh_minutes": 15,
                                           "default_location": "Rangpur",
                                           "notifications": {"daily_summary": True}}).get_json()
    assert s["units"] == "F" and s["notifications"]["daily_summary"] is True
    assert client.get("/api/settings").get_json()["theme"] == "dark"  # persisted per user
    assert svc.run_ttl == 30 * 60             # one shared model cache; the setting drives page reloads
    assert client.get("/api/dashboard").get_json()["location"]["name"] == "Rangpur"
    html = client.get("/settings").get_data(as_text=True)
    assert 'data-theme="dark"' in html                               # theme applied server-side


def test_settings_validation(client):
    assert client.post("/api/settings", json={"refresh_minutes": 5}).status_code == 400
    assert client.post("/api/settings", json={"theme": "neon"}).status_code == 400
    assert client.post("/api/settings", json=["x"]).status_code == 400


def test_about(client):
    a = client.get("/api/about").get_json()
    assert len(a["models"]) == 6 and a["dataset"]["rows"] > 0
    assert a["disclaimer"].startswith("This is an environmental risk estimate")
