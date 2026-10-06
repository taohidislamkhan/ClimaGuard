"""``flask init-db`` / ``flask create-admin`` / ``flask seed-demo``.

No credentials live in the code: the admin password is typed twice at a
``getpass`` prompt, and demo passwords are random and printed once.
"""

from __future__ import annotations

import json
import sqlite3
from getpass import getpass

import click

from . import storage
from .auth import hash_password
from .extensions import db
from .history import DB_PATH as LEGACY_DB
from .models import User, utcnow
from .security import password_problem, temp_password

DEMO_ACCOUNTS = [
    {"name": "Demo Admin", "email": "demo-admin@example.com", "role": "admin", "profile": None},
    {"name": "Rahim", "email": "rahim@example.com", "role": "user",
     "profile": {"age": 28, "location": "Dhaka", "bmi": 22.5, "activity": "High",
                 "outdoor_exposure": "Moderate", "commute": "bus", "outdoor_hours": 2}},
    {"name": "Karima", "email": "karima@example.com", "role": "user",
     "profile": {"age": 68, "location": "Dhaka", "bmi": 27.0, "activity": "Low",
                 "outdoor_exposure": "High", "asthma": True, "has_cooling": False,
                 "outdoor_hours": 5, "commute": "rickshaw", "consent": True}},
]


def import_legacy(user: User, path) -> list[str]:
    """Move the old single-user ``profile``/``settings`` rows (risk_history.db)
    to ``user``, then drop those tables. Returns what was imported."""
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    con.execute("PRAGMA secure_delete = ON")
    done = []
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, save in (("profile", storage.save_profile), ("settings", storage.save_settings)):
            if table not in tables:
                continue
            row = con.execute(f"SELECT data FROM {table} WHERE id = 1").fetchone()  # noqa: S608 (fixed names)
            if row:
                try:
                    save(user, json.loads(row[0]))
                    done.append(table)
                except (storage.ValidationError, json.JSONDecodeError) as e:
                    click.echo(f"  skipped legacy {table}: {e}")
            con.execute(f"DROP TABLE {table}")
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()
    return done


def register(app) -> None:
    @app.cli.command("init-db")
    def init_db():
        """Create the user database tables (safe to run again)."""
        db.create_all()
        click.echo(f"Database ready: {app.config['SQLALCHEMY_DATABASE_URI']}")

    @app.cli.command("create-admin")
    @click.option("--email", required=True)
    @click.option("--name", required=True)
    @click.option("--import-legacy/--no-import-legacy", "legacy", default=False,
                  help="Move the old single-user profile/settings to this admin.")
    def create_admin(email: str, name: str, legacy: bool):
        """Create an admin account. The password is asked twice (not echoed)."""
        db.create_all()
        email = email.strip().lower()
        if User.query.filter_by(email=email).first():
            raise click.ClickException(f"{email} already exists.")
        pw = getpass("Password: ")
        if problem := password_problem(pw):
            raise click.ClickException(problem)
        if getpass("Repeat password: ") != pw:
            raise click.ClickException("Passwords do not match.")
        user = User(name=name.strip()[:60], email=email, password_hash=hash_password(pw), role="admin")
        db.session.add(user)
        db.session.commit()
        click.echo(f"Admin {email} created.")
        if legacy:
            done = import_legacy(user, LEGACY_DB)
            click.echo(f"Imported legacy {', '.join(done)}." if done else "No legacy profile/settings found.")

    @app.cli.command("seed-demo")
    def seed_demo():
        """Demo accounts for the viva (1 admin + 2 users with fake profiles).
        Running it again resets their passwords."""
        db.create_all()
        click.echo("Demo accounts (shown once; run seed-demo again to reset):")
        for acc in DEMO_ACCOUNTS:
            pw = temp_password()
            user = User.query.filter_by(email=acc["email"]).first()
            if user is None:
                user = User(name=acc["name"], email=acc["email"], password_hash="", created_at=utcnow())
                db.session.add(user)
            user.role, user.is_active, user.must_change_password = acc["role"], True, False
            user.failed_logins, user.locked_until = 0, None
            user.password_hash = hash_password(pw)
            db.session.commit()
            if acc["profile"]:
                storage.save_profile(user, acc["profile"])
            click.echo(f"  {acc['role']:<5}  {acc['email']:<24}  {pw}")
