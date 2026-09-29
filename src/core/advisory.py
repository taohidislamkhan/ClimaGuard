"""Rule-based advisory catalogue (was pipeline/step6).

A disease rule fires when the model's prediction is above a quantile of the
training-period target (thresholds in ``params.yaml`` under ``advisory``).
The overall Medium/High layers follow the classifier's predicted class.
Cardiovascular has no rule: its model has no predictive skill (R² ~ 0).
"""

from __future__ import annotations

from src.utils.config import load_params

ADVISORY_RULES: dict[str, str] = {
    "respiratory_high": (
        "Respiratory risk elevated: wear a mask outdoors, "
        "use air purifiers indoors, avoid outdoor exercise during peak PM2.5 hours."
    ),
    "vector_high": (
        "Vector-borne disease risk elevated: drain stagnant water, "
        "use mosquito nets and repellents, wear long sleeves at dusk."
    ),
    "waterborne_high": (
        "Waterborne risk elevated: boil or treat drinking water, "
        "avoid contact with floodwater, wash hands frequently."
    ),
    "heat_high": (
        "Heat-related risk elevated: stay hydrated, avoid midday exertion, "
        "check on elderly and vulnerable individuals."
    ),
    "overall_medium": (
        "Moderate overall health risk: monitor local advisories, "
        "prepare cooling / clean-air supplies."
    ),
    "overall_high": (
        "High overall health risk: activate emergency public-health protocols, "
        "scale clinic capacity, broadcast protective-behaviour guidance."
    ),
}

THRESHOLDS: dict[str, float] = dict(load_params()["advisory"])
