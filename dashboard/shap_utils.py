"""Local SHAP explanations with friendly, grouped feature labels.

The overall classifier (scaled Logistic Regression) gets exact linear SHAP
against the training mean — coef × standardised x — for the High-class
logit. Tree models get exact TreeSHAP computed per request by XGBoost's
native implementation (Random Forests are converted tree by tree, see
``_forest_shap``), so the web process never imports the shap package.

Engineered columns are grouped under one friendly label (all PM2.5 lags and
rolling windows -> "PM2.5") and their SHAP values summed, which keeps SHAP's
additivity. Only model features can appear; humidity is not one.
"""

from __future__ import annotations

import re
import struct

import numpy as np
import pandas as pd

FRIENDLY = [
    (r"^temp_x_pm25$",                   "Temperature × PM2.5"),
    (r"^pm25_x_aqi$",                    "PM2.5 × Air Quality Index"),
    (r"^rain_x_temp$",                   "Rainfall × Temperature"),
    (r"^pm25_ugm3",                      "PM2.5"),
    (r"^air_quality_index",              "Air Quality Index"),
    (r"^temperature_celsius",            "Temperature"),
    (r"^precipitation_mm",               "Rainfall"),
    (r"^heat_wave_days",                 "Heat-wave days"),
    (r"^waterborne_disease_incidents",   "Recent waterborne cases"),
    (r"^heat_related_admissions",        "Recent heat admissions"),
    (r"^(week|week_sin|week_cos|month)$", "Season (week of year)"),
    (r"^gdp_per_capita_usd$",            "GDP per capita"),
    (r"^healthcare_access_index$",       "Healthcare access"),
    (r"^food_security_index$",           "Food security"),
    (r"^(latitude|longitude)$",          "Geographic location"),
    (r"^(region_|income_level_|climate_zone_)", "Country context"),
]


# Groups built from lags / rolling windows of a disease target itself.
AUTOREGRESSIVE = {"Recent waterborne cases", "Recent heat admissions"}


def friendly_label(feature: str) -> str:
    for pattern, label in FRIENDLY:
        if re.search(pattern, feature):
            return label
    return feature


def _high_index(model) -> int:
    classes = [str(c) for c in getattr(model, "classes_", [])]
    return classes.index("High") if "High" in classes else -1


_UBJ_TYPES = {np.dtype(np.float32): b"d", np.dtype(np.int32): b"l",
              np.dtype(np.int64): b"L", np.dtype(np.uint8): b"U"}


def _ubj(v) -> bytes:
    """Minimal UBJSON writer, in the layout XGBoost's own ``save_raw("ubj")``
    uses. Typed numpy arrays go in as raw big-endian bytes, which makes a
    forest load ~20x faster than through JSON text."""
    if isinstance(v, dict):
        return b"{" + b"".join(_ubj_len(len(k)) + k.encode() + _ubj(x) for k, x in v.items()) + b"}"
    if isinstance(v, str):
        return b"S" + _ubj_len(len(v)) + v.encode()
    if isinstance(v, int):
        return b"L" + struct.pack(">q", v)
    if isinstance(v, np.ndarray):
        return (b"[$" + _UBJ_TYPES[v.dtype] + b"#" + _ubj_len(v.size)
                + v.astype(v.dtype.newbyteorder(">")).tobytes())
    return b"[#" + _ubj_len(len(v)) + b"".join(_ubj(x) for x in v)      # list


def _ubj_len(n: int) -> bytes:
    return b"L" + struct.pack(">q", n)


def _xgb_tree(i: int, tree, scale: float) -> dict:
    """One fitted sklearn tree as an XGBoost tree (leaf values × ``scale``).

    sklearn sends a float32 ``x`` left when ``x <= threshold``; XGBoost when
    ``x < split``. Rounding each threshold up to the next float32 above it
    makes the two tests agree for every float32 input.
    """
    n = tree.node_count
    left, right = tree.children_left, tree.children_right
    leaf = left == -1
    split = tree.threshold.astype(np.float32)
    split = np.where(split <= tree.threshold, np.nextafter(split, np.float32(np.inf)), split)
    value = (tree.value[:, 0, 0] * scale).astype(np.float32)
    parents = np.full(n, 2147483647, dtype=np.int32)
    parents[left[~leaf]] = np.flatnonzero(~leaf)
    parents[right[~leaf]] = np.flatnonzero(~leaf)
    none32, none64, zeros = np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(n, np.uint8)
    return {
        "base_weights": value, "categories": none32, "categories_nodes": none32,
        "categories_segments": none64, "categories_sizes": none64, "default_left": zeros, "id": i,
        "left_children": left.astype(np.int32), "loss_changes": np.zeros(n, np.float32),
        "parents": parents, "right_children": right.astype(np.int32),
        "split_conditions": np.where(leaf, value, split).astype(np.float32),
        "split_indices": np.where(leaf, 0, tree.feature).astype(np.int32), "split_type": zeros,
        "sum_hessian": tree.weighted_n_node_samples.astype(np.float32),   # cover, as shap uses
        "tree_param": {"num_deleted": "0", "num_feature": str(tree.n_features),
                       "num_nodes": str(n), "size_leaf_vector": "1"},
    }


def _forest_shap(forest, x: pd.DataFrame, chunk: int = 50) -> np.ndarray:
    """Exact path-dependent TreeSHAP of a sklearn Random Forest regressor.

    The same algorithm as ``shap.TreeExplainer`` (cover = weighted node
    samples), run by XGBoost's native implementation; results agree to
    float32 precision. SHAP is additive over trees, so the forest goes
    through in chunks and only a few MB are ever in use: neither the shap
    package (~50 MB with numba) nor a full explainer (~30 MB per forest) is
    loaded into the web process.
    """
    import xgboost   # already loaded: the heat model is XGBoost
    X = xgboost.DMatrix(np.asarray(x, dtype=np.float32))
    trees = forest.estimators_
    version = [int(v) for v in xgboost.__version__.split(".")[:3]]
    out = np.zeros((X.num_row(), X.num_col()))
    for start in range(0, len(trees), chunk):
        part = [_xgb_tree(i, t.tree_, 1.0 / len(trees))
                for i, t in enumerate(trees[start:start + chunk])]
        raw = _ubj({"learner": {
            "attributes": {}, "feature_names": [], "feature_types": [],
            "gradient_booster": {"name": "gbtree", "model": {
                "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": str(len(part))},
                "iteration_indptr": list(range(len(part) + 1)), "tree_info": [0] * len(part),
                "trees": part}},
            "learner_model_param": {"base_score": "0", "boost_from_average": "0", "num_class": "0",
                                    "num_feature": str(X.num_col()), "num_target": "1"},
            "objective": {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}}},
            "version": version})
        del part
        booster = xgboost.Booster()
        booster.load_model(bytearray(raw))
        del raw
        out += booster.predict(X, pred_contribs=True)[:, :-1]   # last column is the bias
        del booster
    return out


def tree_shap(model, x: pd.DataFrame) -> np.ndarray:
    """Exact (path-dependent) TreeSHAP for the rows of ``x``, computed on demand."""
    if hasattr(model, "get_booster"):
        import xgboost
        contribs = model.get_booster().predict(xgboost.DMatrix(x), pred_contribs=True)
        return np.asarray(contribs)[:, :-1]          # last column is the bias term
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "tree_"):
        return _forest_shap(model, x)
    from shap import TreeExplainer   # any other tree model: fall back to shap itself
    return TreeExplainer(model).shap_values(x)


def local_shap(model, x: pd.DataFrame) -> np.ndarray:
    """Per-feature SHAP values for a single-row frame ``x``."""
    steps = getattr(model, "steps", None)
    final = steps[-1][1] if steps else getattr(model, "model", model)   # unwrap EncodedClassifier
    if hasattr(final, "coef_") and steps:
        z = np.asarray(model[:-1].transform(x), dtype=float)[0]
        coef = np.asarray(final.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[_high_index(model)] if coef.shape[0] > 1 else coef[0]
        return coef * z
    sv = tree_shap(final, x)
    if isinstance(sv, list):
        return np.asarray(sv[_high_index(model)])[0]
    arr = np.asarray(sv)
    return arr[0, :, _high_index(model)] if arr.ndim == 3 else arr[0]


def contribution_level(norm: float) -> str:
    if norm >= 66:
        return "High"
    if norm >= 33:
        return "Moderate"
    return "Low"


def grouped_signed(model, x: pd.DataFrame, k: int = 8) -> list[dict]:
    """Top-k grouped factors with signed SHAP values (waterfall-style charts)."""
    sv = local_shap(model, x)
    df = pd.DataFrame({"feature": x.columns, "shap": sv})
    df["label"] = df["feature"].map(friendly_label)
    g = (df.groupby("label").agg(shap=("shap", "sum"), features=("feature", list))
           .reset_index())
    g = g[g["shap"].abs() > 1e-9]                  # linear models zero out many features
    g = g.reindex(g["shap"].abs().sort_values(ascending=False).index).head(k)
    return [{"label": r["label"], "shap": round(float(r["shap"]), 4), "features": r["features"],
             "autoregressive": r["label"] in AUTOREGRESSIVE} for _, r in g.iterrows()]


def top_factors(model, x: pd.DataFrame, k: int = 6) -> list[dict]:
    """Top-k grouped factors, bar length = |SHAP| normalised to 0–100."""
    sv = local_shap(model, x)
    df = pd.DataFrame({"feature": x.columns, "shap": sv})
    df["label"] = df["feature"].map(friendly_label)
    grouped = (df.groupby("label")
                 .agg(shap=("shap", "sum"), features=("feature", list))
                 .reset_index())
    grouped["abs"] = grouped["shap"].abs()
    grouped = grouped.sort_values("abs", ascending=False).head(k)
    top = float(grouped["abs"].max()) or 1.0
    out = []
    for _, r in grouped.iterrows():
        norm = round(100.0 * r["abs"] / top)
        out.append({
            "label": r["label"],
            "value": norm,
            "shap": float(r["shap"]),
            "direction": "raises" if r["shap"] > 0 else "lowers",
            "level": contribution_level(norm),
            "features": r["features"],
            "autoregressive": r["label"] in AUTOREGRESSIVE,
        })
    return out
