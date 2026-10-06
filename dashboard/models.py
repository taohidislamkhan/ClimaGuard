"""User accounts, per-user profile/settings and the audit log (``instance/app.db``).

This database is serving state only: git-ignored, not DVC-tracked, and never
read by the ML pipeline. ``risk_history`` stays in ``data/cache`` because it
is regional, not per user.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from flask_login import UserMixin
from sqlalchemy import event
from sqlalchemy.engine import Engine

from .extensions import db

ROLES = ("user", "admin")


def utcnow() -> datetime:
    """Naive UTC (SQLite stores datetimes without a zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_session_token() -> str:
    return secrets.token_hex(16)


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    """Deleted rows (health data) are overwritten on disk; FKs are enforced."""
    if type(dbapi_conn).__module__.startswith("sqlite3"):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA secure_delete = ON")
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), nullable=False, default="user")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    must_change_password = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    last_login_at = db.Column(db.DateTime)
    failed_logins = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime)
    # Rotated on password change/reset and deactivation: every existing
    # session and remember-me cookie for this user stops working.
    session_token = db.Column(db.String(32), nullable=False, default=new_session_token, index=True)

    profile = db.relationship("Profile", uselist=False, cascade="all, delete-orphan", passive_deletes=True)
    settings = db.relationship("UserSettings", uselist=False, cascade="all, delete-orphan", passive_deletes=True)

    def get_id(self) -> str:
        return self.session_token

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def is_locked(self, now: datetime | None = None) -> bool:
        return bool(self.locked_until and self.locked_until > (now or utcnow()))

    def account_dict(self) -> dict:
        """Account metadata only: what admin pages may see (no health data)."""
        iso = lambda d: d.isoformat(timespec="seconds") + "Z" if d else None  # noqa: E731
        return {"id": self.id, "name": self.name, "email": self.email, "role": self.role,
                "is_active": self.is_active, "locked": self.is_locked(),
                "must_change_password": self.must_change_password,
                "created_at": iso(self.created_at), "last_login_at": iso(self.last_login_at)}


class Profile(db.Model):
    __tablename__ = "profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"),
                        unique=True, nullable=False)
    data = db.Column(db.JSON, nullable=False)          # validated by storage.clean_profile
    health_consent = db.Column(db.Boolean, nullable=False, default=False)
    consent_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class UserSettings(db.Model):
    __tablename__ = "settings"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"),
                        unique=True, nullable=False)
    data = db.Column(db.JSON, nullable=False)          # validated by storage.clean_settings


class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = db.Column(db.Integer, primary_key=True)
    actor_user_id = db.Column(db.Integer, index=True)  # no FK: entries outlive deleted users
    action = db.Column(db.String(40), nullable=False, index=True)
    target = db.Column(db.String(300))
    timestamp = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    ip = db.Column(db.String(45))

    def as_dict(self) -> dict:
        return {"id": self.id, "actor_user_id": self.actor_user_id, "action": self.action,
                "target": self.target, "timestamp": self.timestamp.isoformat(timespec="seconds") + "Z",
                "ip": self.ip}
