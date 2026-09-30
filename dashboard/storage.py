"""Single-user profile and settings, persisted in SQLite.

Both live in the same database file as ``risk_history`` as one JSON
document each (tables ``profile`` and ``settings``, row id 1). Unknown keys
are dropped and values are validated before they are stored.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .history import DB_PATH

DIVISION_NAMES = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Sylhet",
                  "Barisal", "Rangpur", "Mymensingh"]
LEVELS = ["Low", "Moderate", "High"]

DEFAULT_PROFILE = {
    "name": "User", "age": 22, "location": "Dhaka",
    "height_cm": None, "weight_kg": None, "bmi": 21.4,
    "activity": "Moderate", "outdoor_exposure": "High", "smoking": False,
    "asthma": False, "cardiovascular_disease": False, "diabetes": False,
}

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
    for k in ("smoking", "asthma", "cardiovascular_disease", "diabetes"):
        p[k] = bool(p.get(k))
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


class AppStore:
    def __init__(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            for table in ("profile", "settings"):
                self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} "
                                  "(id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL)")
            self.conn.commit()

    def _get(self, table: str, default: dict, cleaner) -> dict:
        with self._lock:
            row = self.conn.execute(f"SELECT data FROM {table} WHERE id = 1").fetchone()
        if not row:
            return dict(default) if table == "profile" else json.loads(json.dumps(default))
        try:
            return cleaner(json.loads(row[0]))
        except (ValidationError, json.JSONDecodeError):
            return json.loads(json.dumps(default))

    def _put(self, table: str, data: dict) -> None:
        with self._lock:
            self.conn.execute(f"INSERT OR REPLACE INTO {table} (id, data) VALUES (1, ?)",
                              (json.dumps(data),))
            self.conn.commit()

    def profile(self) -> dict:
        return self._get("profile", DEFAULT_PROFILE, clean_profile)

    def save_profile(self, data: dict) -> dict:
        p = clean_profile(data, self.profile())
        self._put("profile", p)
        return p

    def settings(self) -> dict:
        return self._get("settings", DEFAULT_SETTINGS, clean_settings)

    def save_settings(self, data: dict) -> dict:
        s = clean_settings(data, self.settings())
        self._put("settings", s)
        return s
