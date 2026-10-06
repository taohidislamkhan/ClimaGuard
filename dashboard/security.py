"""Access control and web-security helpers shared by the app and blueprints."""

from __future__ import annotations

import re
import secrets
from functools import wraps
from urllib.parse import urlsplit

from flask import abort, g, request
from flask_login import current_user, login_required

from .extensions import db
from .models import AuditLog

__all__ = ["login_required", "role_required", "safe_next", "password_problem", "audit",
           "client_ip", "temp_password", "set_security_headers", "csp_nonce"]

PASSWORD_RULE = "At least 8 characters, with at least one letter and one number."

# Third-party origins the templates already load from (see templates/*.html).
_CSP = ("default-src 'self'; "
        "script-src 'self' 'nonce-{nonce}' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'")


def role_required(role: str):
    """Allow only signed-in users with ``role``. Guests get 401 (JSON for
    /api/*, a login redirect for pages); other roles get 403."""
    def deco(fn):
        @wraps(fn)
        @login_required
        def wrapper(*args, **kwargs):
            if current_user.role != role:
                abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return deco


def safe_next(target: str | None, default: str = "/") -> str:
    """Only same-site relative paths ("/my-health?loc=Sylhet"); anything with a
    scheme or host, protocol-relative ("//evil.com") or backslash tricks falls
    back to ``default``."""
    if not target or not isinstance(target, str):
        return default
    target = target.strip()
    if "\\" in target or any(ord(c) < 32 for c in target):
        return default
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return default
    return target


def password_problem(pw: str) -> str | None:
    """None if ``pw`` follows the rule, else a message."""
    if len(pw or "") < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        return PASSWORD_RULE
    return None


def temp_password() -> str:
    """Random temporary password that satisfies the rule (shown once to the admin)."""
    while True:
        pw = secrets.token_urlsafe(9)
        if not password_problem(pw):
            return pw


def client_ip() -> str | None:
    return (request.remote_addr or "")[:45] or None


def audit(action: str, target: str | None = None, actor_id: int | None = None) -> None:
    """Append to the audit log. Callers pass ids/emails only, never passwords
    or profile fields."""
    if actor_id is None and current_user and current_user.is_authenticated:
        actor_id = current_user.id
    db.session.add(AuditLog(actor_user_id=actor_id, action=action,
                            target=(target or "")[:300] or None, ip=client_ip()))
    db.session.commit()


def csp_nonce() -> str:
    if "csp_nonce" not in g:
        g.csp_nonce = secrets.token_urlsafe(16)
    return g.csp_nonce


def set_security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Content-Security-Policy", _CSP.format(nonce=csp_nonce()))
    return resp
