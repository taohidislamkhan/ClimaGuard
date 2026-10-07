import numpy as np
import pandas as pd
import pytest

from dashboard import advisory, scoring, shap_utils
from dashboard.inference import WEATHER_COLS, derive_weather_features
from dashboard.weather_client import Climatology


@pytest.mark.parametrize("score,level", [(0, "Low"), (39.9, "Low"), (40, "Moderate"),
                                         (64.9, "Moderate"), (65, "High"), (100, "High")])
def test_band_edges(score, level):
    assert scoring.band(score) == level


def test_overall_score_formula():
    assert scoring.overall_score({"Low": 1, "Medium": 0, "High": 0}) == 0
    assert scoring.overall_score({"Low": 0, "Medium": 1, "High": 0}) == 50
    assert scoring.overall_score({"Low": 0, "Medium": 0, "High": 1}) == 100
    assert scoring.overall_score({"Low": .2, "Medium": .4, "High": .4}) == pytest.approx(60)


def test_overall_level_is_argmax_and_renamed():
    assert scoring.overall_level({"Low": .1, "Medium": .6, "High": .3}) == "Moderate"
    assert scoring.overall_level({"Low": .1, "Medium": .2, "High": .7}) == "High"


def test_adjustments_capped_at_ten():
    p = {"asthma": True, "outdoor_exposure": "High", "smoking": True, "age": 70,
         "heart_condition": True}
    adj = scoring.personal_adjustments(p)
    assert adj["respiratory"]["points"] == 10          # 5 + 5 + 3 capped
    assert adj["cardio"]["points"] == 10                # 5 + 3 + 5 capped
    assert adj["heat"]["points"] == 8
    assert len(adj["respiratory"]["reasons"]) == 3


def test_no_adjustment_for_neutral_profile():
    assert scoring.personal_adjustments({"age": 30, "outdoor_exposure": "Low"}) == {}


def test_apply_adjustment_clamps():
    assert scoring.apply_adjustment(97, 10) == 100
    assert scoring.apply_adjustment(3, -10) == 0


def test_pct_change():
    assert scoring.pct_change(110, 100) == pytest.approx(10)
    assert scoring.pct_change(5, 0) is None
    assert scoring.pct_change(5, None) is None


def test_live_feature_derivation_matches_pipeline(store):
    """Recomputing step-3 weather features from base columns reproduces the dataset."""
    hist = store.country.tail(13).reset_index(drop=True)
    derived = derive_weather_features(hist[WEATHER_COLS])
    last = hist.iloc[-1]
    checked = 0
    for feat in store.features:
        if feat in derived:
            assert derived[feat] == pytest.approx(float(last[feat]), rel=1e-6, abs=1e-6), feat
            checked += 1
    assert checked >= 25


def test_percentile_monotonic(store):
    arr = store.train_targets["respiratory"]
    lo, mid, hi = np.quantile(arr, [0.1, 0.5, 0.9])
    assert store.percentile("respiratory", lo) < store.percentile("respiratory", mid) \
        < store.percentile("respiratory", hi)
    assert store.percentile("respiratory", arr.max() + 1) == 100


def test_quantile_map_monotonic_and_invertible():
    rng = np.random.default_rng(0)
    clim = Climatology(np.sort(rng.normal(26, 4, 500)), np.sort(rng.normal(11, 6, 500)), 35)
    mapped = [clim.to_model_temp(t) for t in (18, 24, 28, 32)]
    assert mapped == sorted(mapped)
    assert clim.to_real_temp(clim.to_model_temp(27)) == pytest.approx(27, abs=0.5)


def test_shap_excludes_non_model_features(store):
    x = store.to_matrix([store.template])
    factors = shap_utils.top_factors(store.models["overall"], x, k=50)
    labels = {f["label"] for f in factors}
    assert "Humidity" not in labels
    for f in factors:
        assert set(f["features"]) <= set(store.features)
    assert max(f["value"] for f in factors) == 100


def test_linear_shap_is_additive(store):
    """coef·z summed + intercept equals the High-class logit."""
    model = store.models["overall"]
    x = store.to_matrix([store.template])
    sv = shap_utils.local_shap(model, x)
    clf = model[-1]
    hi = list(clf.classes_).index("High")
    logit = model.decision_function(x)[0][hi]
    assert sv.sum() + clf.intercept_[hi] == pytest.approx(logit, rel=1e-6)



@pytest.mark.parametrize("key", ["respiratory", "vector", "waterborne", "heat"])
def test_tree_shap_matches_shap_package(store, key):
    """The web app's native TreeSHAP equals shap.TreeExplainer (float32 precision)
    and adds up to the prediction; the shap package itself stays out of the app."""
    from shap import TreeExplainer
    model = store.models[key]
    x = store.to_matrix(store.country_history(3))
    ours = shap_utils.tree_shap(model, x)
    ref = np.asarray(TreeExplainer(model).shap_values(x))
    assert np.abs(ours - ref).max() <= 1e-5 * np.abs(ref).max()
    base = model.predict(x) - ref.sum(1)
    assert ours.sum(1) + base == pytest.approx(model.predict(x), rel=1e-5, abs=1e-4)


def test_models_load_lazily():
    from dashboard.inference import ModelStore
    st = ModelStore()
    assert dict(st.models) == {} and st._data is None
    st.predict(st.to_matrix([st.template]))
    assert set(st.models) == {"overall", "respiratory", "vector", "heat", "waterborne", "cardio"}


def test_friendly_labels():
    assert shap_utils.friendly_label("pm25_ugm3_roll_mean_4w") == "PM2.5"
    assert shap_utils.friendly_label("temperature_celsius_lag8w") == "Temperature"
    assert shap_utils.friendly_label("week_sin") == "Season (week of year)"


def test_advisories_follow_engine_thresholds():
    scores = {"respiratory": 85, "heat": 10, "vector": 10, "waterborne": 79}
    adv = advisory.build_advisories("Medium", {"Low": .1, "Medium": .7, "High": .2}, scores, 80)
    keys = [a["key"] for a in adv]
    assert "respiratory_high" in keys and "waterborne_high" not in keys
    assert "overall_medium" in keys
    assert keys[0] == "respiratory_high"                       # most severe first
    assert all(len(a["bullets"]) <= 3 for a in adv)


def test_routine_advisory_when_nothing_fires():
    scores = {"respiratory": 10, "heat": 10, "vector": 10, "waterborne": 10}
    adv = advisory.build_advisories("Low", {"Low": .9, "Medium": .1, "High": 0}, scores, 5)
    assert [a["key"] for a in adv] == ["routine"]


def test_compare_label():
    from dashboard.service import _compare_label
    assert _compare_label(None, "2026-09-27") == "no earlier snapshot"
    assert _compare_label({"source": "backfill", "date": "2025-10-19"}, "2026-09-27") == "vs last week (dataset)"
    assert _compare_label({"source": "live", "date": "2026-09-26"}, "2026-09-27") == "vs yesterday"
    assert _compare_label({"source": "live", "date": "2026-09-20"}, "2026-09-27") == "vs last snapshot"
