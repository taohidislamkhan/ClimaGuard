"""Production entry point (Render / any gunicorn host)::

    gunicorn --workers 1 --threads 4 --bind 0.0.0.0:$PORT wsgi:app

One worker on purpose: the rate limiter counts in memory and accounts live in
SQLite, so a single process keeps both consistent.

Optional first-admin bootstrap: set ADMIN_EMAIL and ADMIN_PASSWORD in the host's
environment and the account is created at startup if it does not exist yet.
That covers hosts with no shell (and free plans whose disk is wiped on restart).
"""

from __future__ import annotations

import os

from werkzeug.middleware.proxy_fix import ProxyFix

from app import create_app, start_warmup
from dashboard.auth import hash_password
from dashboard.extensions import db
from dashboard.models import User
from dashboard.security import password_problem

app = create_app()
# Render terminates HTTPS at its proxy: trust one hop for the client IP (rate
# limits, audit log) and the scheme (secure redirects).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)


def _bootstrap_admin() -> None:
    email = os.environ.get("ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not email or not password:
        return
    with app.app_context():
        if User.query.filter_by(email=email).first():
            return
        if problem := password_problem(password):
            raise RuntimeError(f"ADMIN_PASSWORD rejected: {problem}")
        name = os.environ.get("ADMIN_NAME", "Admin").strip()[:60] or "Admin"
        db.session.add(User(name=name, email=email, password_hash=hash_password(password), role="admin"))
        db.session.commit()
        app.logger.warning("Bootstrap admin %s created.", email)


_bootstrap_admin()
start_warmup(app)
