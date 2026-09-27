"""Score conversion, risk bands, and the rule-based personal adjustment.

All of these are deterministic post-processing of the model outputs; the
profile never enters the ML models.
"""

from __future__ import annotations

LOW, MODERATE, HIGH = "Low", "Moderate", "High"
ADJUST_CAP = 10


def band(score: float) -> str:
    """Disease score bands: <40 Low, 40–64 Moderate, >=65 High."""
    if score >= 65:
        return HIGH
    if score >= 40:
        return MODERATE
    return LOW


def overall_score(proba: dict[str, float]) -> float:
    """100 × (0.5·P(Medium) + 1.0·P(High)) from the overall classifier."""
    return 100.0 * (0.5 * proba.get("Medium", 0.0) + proba.get("High", 0.0))


def overall_level(proba: dict[str, float]) -> str:
    """Argmax class, with the classifier's "Medium" shown as "Moderate"."""
    cls = max(proba, key=proba.get)
    return MODERATE if cls == "Medium" else cls


def pct_change(new: float | None, old: float | None) -> float | None:
    """Relative change in percent; None when undefined."""
    if new is None or old is None or old == 0:
        return None
    return 100.0 * (new - old) / abs(old)


# ---------------------------------------------------------------------------
# Personal adjustment (applied AFTER the model, capped at ±10 per disease)
# ---------------------------------------------------------------------------
# (condition description, predicate(profile), {disease: points})
ADJUSTMENT_RULES = [
    ("Asthma",                 lambda p: bool(p.get("asthma")),                 {"respiratory": 5}),
    ("High outdoor exposure",  lambda p: p.get("outdoor_exposure") == "High",   {"respiratory": 5, "heat": 3}),
    ("Age 65 or over",         lambda p: (p.get("age") or 0) >= 65,             {"heat": 5, "cardio": 5}),
    ("Smoker",                 lambda p: bool(p.get("smoking")),                {"respiratory": 3, "cardio": 3}),
    ("Cardiovascular disease", lambda p: bool(p.get("cardiovascular_disease") or p.get("heart_condition")),
                                                                                {"cardio": 5}),
    ("Diabetes",               lambda p: bool(p.get("diabetes")),               {"cardio": 3, "heat": 2}),
    ("BMI 30 or over",         lambda p: (p.get("bmi") or 0) >= 30,             {"heat": 3, "cardio": 2}),
]


def rule_table(profile: dict) -> list[dict]:
    """Every rule with whether it fires for ``profile`` (for the live preview)."""
    return [{"label": label, "fires": bool(pred(profile)), "effects": effects}
            for label, pred, effects in ADJUSTMENT_RULES]


def personal_adjustments(profile: dict) -> dict[str, dict]:
    """Per-disease {points, raw_points, capped, reasons[]} from the profile, capped at ±10."""
    out: dict[str, dict] = {}
    for label, pred, effects in ADJUSTMENT_RULES:
        if not pred(profile):
            continue
        for disease, pts in effects.items():
            entry = out.setdefault(disease, {"points": 0, "reasons": []})
            entry["points"] += pts
            entry["reasons"].append(f"{label} (+{pts})")
    for entry in out.values():
        entry["raw_points"] = entry["points"]
        entry["points"] = max(-ADJUST_CAP, min(ADJUST_CAP, entry["points"]))
        entry["capped"] = entry["points"] != entry["raw_points"]
    return out


def apply_adjustment(score: float, points: int) -> float:
    return max(0.0, min(100.0, score + points))
