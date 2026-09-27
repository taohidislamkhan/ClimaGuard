"""Wrap the pipeline's rule-based advisory engine for the dashboard.

Trigger logic and message text come from ``pipeline/step6_advisory.py``
(``ADVISORY_RULES`` / ``THRESHOLDS``). A disease rule fires when the
model's prediction clears the configured quantile of the training target —
equivalently, when its 0–100 percentile score reaches ``q × 100``. Overall
Medium/High layers follow the classifier's argmax class. This module only
adds display metadata (title, icon, tone) and splits each rule's actions
into bullets.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from step6_advisory import ADVISORY_RULES, THRESHOLDS  # noqa: E402

DISCLAIMER = "This is an environmental risk estimate, not a medical diagnosis."
FOOTER = ("General environmental health guidance. For medical concerns, "
          "consult a qualified healthcare professional.")

DISPLAY = {
    "respiratory_high": {"title": "Air Quality Alert", "icon": "wind", "tone": "high",
                         "reason": "Respiratory risk model is in the top {pct} of training weeks (PM2.5 {pm25} µg/m³)."},
    "heat_high":        {"title": "Heat Advisory", "icon": "sun", "tone": "moderate",
                         "reason": "Heat-related risk model is in the top {pct} of training weeks."},
    "vector_high":      {"title": "Vector-borne Alert", "icon": "bug", "tone": "moderate",
                         "reason": "Vector-borne risk model is in the top {pct} of training weeks."},
    "waterborne_high":  {"title": "Waterborne Alert", "icon": "droplet", "tone": "moderate",
                         "reason": "Waterborne risk model is in the top {pct} of training weeks."},
    "overall_high":     {"title": "High Overall Risk", "icon": "triangle-alert", "tone": "high",
                         "reason": "Overall classifier predicts High (P = {p:.0%})."},
    "overall_medium":   {"title": "Moderate Overall Risk", "icon": "info", "tone": "moderate",
                         "reason": "Overall classifier predicts Medium (P = {p:.0%})."},
}

DISEASE_RULES = [("respiratory", "respiratory_q", "respiratory_high"),
                 ("heat", "heat_q", "heat_high"),
                 ("vector", "vector_q", "vector_high"),
                 ("waterborne", "waterborne_q", "waterborne_high")]


def _bullets(rule_text: str) -> list[str]:
    """'Header: a, b, c.' -> ['A', 'B', 'C']"""
    actions = rule_text.split(":", 1)[-1].strip().rstrip(".")
    return [a.strip()[:1].upper() + a.strip()[1:] for a in actions.split(",") if a.strip()]


def build_advisories(overall_class: str, proba: dict[str, float],
                     model_scores: dict[str, float], pm25: float | None) -> list[dict]:
    """Advisories for the current prediction, most severe first."""
    out = []
    for disease, q_key, rule in DISEASE_RULES:
        q = THRESHOLDS[q_key]
        if model_scores[disease] >= q * 100:
            meta = DISPLAY[rule]
            out.append({
                "key": rule, "title": meta["title"], "icon": meta["icon"], "tone": meta["tone"],
                "reason": meta["reason"].format(pct=f"{round((1 - q) * 100)}%",
                                                pm25="–" if pm25 is None else f"{pm25:.0f}"),
                "bullets": _bullets(ADVISORY_RULES[rule])[:3],
            })
    layer = {"High": "overall_high", "Medium": "overall_medium"}.get(overall_class)
    if layer:
        meta = DISPLAY[layer]
        out.append({
            "key": layer, "title": meta["title"], "icon": meta["icon"], "tone": meta["tone"],
            "reason": meta["reason"].format(p=proba[overall_class]),
            "bullets": _bullets(ADVISORY_RULES[layer])[:3],
        })
    if not out:
        out.append({"key": "routine", "title": "No active alerts", "icon": "circle-check",
                    "tone": "low", "reason": "No disease trigger cleared its threshold.",
                    "bullets": ["Routine surveillance", "Check back after the next update"]})
    order = {"high": 0, "moderate": 1, "low": 2}
    return sorted(out, key=lambda a: order[a["tone"]])
