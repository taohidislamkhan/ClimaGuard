"""Per-user profile and settings (``profiles`` / ``settings`` in ``instance/app.db``).

Each user has at most one profile and one settings row, stored as a validated
JSON document. Unknown keys are dropped and values are validated before they
are stored. Every read and write takes the *current user*: there is no way to
pass another user's id in.

Privacy: the profile is never logged, never shown to admins and never sent to
an external API (Open-Meteo only receives division coordinates). Saving a
health condition requires ``consent``; deleting the profile removes the row
(``secure_delete`` overwrites the freed pages).
"""

from __future__ import annotations

import json

from .extensions import db
from .models import Profile, UserSettings, utcnow

DIVISION_NAMES = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Sylhet",
                  "Barisal", "Rangpur", "Mymensingh"]
LEVELS = ["Low", "Moderate", "High"]

DEFAULT_PROFILE = {
    "name": "User", "age": 22, "location": "Dhaka",
    "height_cm": None, "weight_kg": None, "bmi": 21.4,
    "activity": "Moderate", "outdoor_exposure": "High", "smoking": False,
    "asthma": False, "cardiovascular_disease": False, "diabetes": False,
    # personal advisory fields (dashboard/personal_rules.py)
    "pregnancy": False, "weakened_immunity": False,
    "outdoor_work": False, "outdoor_hours": 2, "commute": "bus",
    "has_cooling": True, "mosquito_nets": True, "water_source": "filtered",
    "consent": False,
}

# Health conditions that need the consent box ticked before they are saved.
CONDITIONS = ["asthma", "cardiovascular_disease", "diabetes", "pregnancy", "weakened_immunity"]
COMMUTES = ["walk", "rickshaw", "bus", "car"]
WATER_SOURCES = ["tap", "filtered", "boiled", "tube_well"]

DEFAULT_SETTINGS = {
    "units": "C", "theme": "light", "default_location": "Dhaka",
    "refresh_minutes": 30,
    "notifications": {"any_high": True, "daily_summary": False, "air_quality": True},
}


class ValidationError(ValueError):
    pass


def _num(v, lo, hi, name, allow_none=False):
    if v in (None, ""):
        if allow_none:
            return None
        raise ValidationError(f"{name} is required")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number") from None
    if not lo <= x <= hi:
        raise ValidationError(f"{name} must be between {lo} and {hi}")
    return x


def clean_profile(data: dict, base: dict | None = None) -> dict:
    """Merge ``data`` over ``base`` and validate. Legacy ``heart_condition``
    maps to ``cardiovascular_disease``. BMI is computed from height and
    weight when both are given."""
    p = dict(base or DEFAULT_PROFILE)
    data = dict(data)
    if "heart_condition" in data and "cardiovascular_disease" not in data:
        data["cardiovascular_disease"] = data.pop("heart_condition")
    for k, v in data.items():
        if k in DEFAULT_PROFILE:
            p[k] = v
    p["name"] = str(p.get("name") or "").strip()[:30] or "Guest"
    p["age"] = int(_num(p.get("age"), 1, 110, "Age"))
    if p.get("location") not in DIVISION_NAMES:
        raise ValidationError("Unknown location")
    for k in ("activity", "outdoor_exposure"):
        if p.get(k) not in LEVELS:
            raise ValidationError(f"{k} must be Low, Moderate or High")
    for k in ("smoking", *CONDITIONS, "outdoor_work", "has_cooling", "mosquito_nets", "consent"):
        p[k] = bool(p.get(k))
    p["outdoor_hours"] = _num(p.get("outdoor_hours"), 0, 24, "Outdoor hours")
    if p.get("commute") not in COMMUTES:
        raise ValidationError(f"commute must be one of {COMMUTES}")
    if p.get("water_source") not in WATER_SOURCES:
        raise ValidationError(f"water_source must be one of {WATER_SOURCES}")
    p["height_cm"] = _num(p.get("height_cm"), 50, 250, "Height", allow_none=True)
    p["weight_kg"] = _num(p.get("weight_kg"), 10, 300, "Weight", allow_none=True)
    if p["height_cm"] and p["weight_kg"]:
        p["bmi"] = round(p["weight_kg"] / (p["height_cm"] / 100) ** 2, 1)
    else:
        p["bmi"] = _num(p.get("bmi"), 10, 70, "BMI", allow_none=True)
        p["bmi"] = None if p["bmi"] is None else round(p["bmi"], 1)
    return p


def clean_settings(data: dict, base: dict | None = None) -> dict:
    s = json.loads(json.dumps(base or DEFAULT_SETTINGS))
    for k, v in data.items():
        if k == "notifications" and isinstance(v, dict):
            for nk, nv in v.items():
                if nk in s["notifications"]:
                    s["notifications"][nk] = bool(nv)
        elif k in DEFAULT_SETTINGS:
            s[k] = v
    if s["units"] not in ("C", "F"):
        raise ValidationError("units must be C or F")
    if s["theme"] not in ("light", "dark"):
        raise ValidationError("theme must be light or dark")
    if s["default_location"] not in DIVISION_NAMES:
        raise ValidationError("Unknown default location")
    try:
        s["refresh_minutes"] = int(s["refresh_minutes"])
    except (TypeError, ValueError):
        raise ValidationError("refresh_minutes must be 15, 30 or 60") from None
    if s["refresh_minutes"] not in (15, 30, 60):
        raise ValidationError("refresh_minutes must be 15, 30 or 60")
    return s


# -- per-user persistence (always scoped to the given user) --------------------
def has_profile(user) -> bool:
    return user is not None and user.profile is not None


def profile_for(user) -> dict | None:
    """The user's saved profile (name from the account), or None when not saved."""
    if not has_profile(user):
        return None
    try:
        p = clean_profile(user.profile.data)
    except ValidationError:
        p = dict(DEFAULT_PROFILE)
    p["name"] = user.name
    return p


def profile_or_default(user) -> dict:
    return profile_for(user) or {**DEFAULT_PROFILE, "name": user.name if user else "Guest"}


def save_profile(user, data: dict) -> dict:
    data = {k: v for k, v in data.items() if k != "name"}   # the name lives on the account
    p = clean_profile(data, profile_or_default(user))
    if not p["consent"] and any(p[k] for k in CONDITIONS):
        raise ValidationError("Tick the consent box before saving a health condition")
    p["name"] = user.name
    row = user.profile or Profile(user_id=user.id, data={})
    if p["consent"] and not row.health_consent:
        row.consent_at = utcnow()
    row.health_consent = p["consent"]
    if not p["consent"]:
        row.consent_at = None
    row.data = p
    db.session.add(row)
    db.session.commit()
    return p


def delete_profile(user) -> None:
    if user.profile is not None:
        db.session.delete(user.profile)
        db.session.commit()
        db.session.refresh(user)


def settings_for(user) -> dict:
    if user is None or user.settings is None:
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        return clean_settings(user.settings.data)
    except ValidationError:
        return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_settings(user, data: dict) -> dict:
    s = clean_settings(data, settings_for(user))
    row = user.settings or UserSettings(user_id=user.id, data={})
    row.data = s
    db.session.add(row)
    db.session.commit()
    return s
