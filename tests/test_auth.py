"""Sign up / log in / log out, passwords, lockout, redirects, CSRF, sessions, CLI."""

import re
import sqlite3
from datetime import timedelta

import pytest
from werkzeug.security import check_password_hash

from conftest import PASSWORD
from dashboard.models import AuditLog, User, utcnow
from dashboard.security import safe_next


def user_row(app, email):
    with app.app_context():
        return User.query.filter_by(email=email).first()


def signup(c, email="new@example.com", password="goodpass1", confirm=None, **extra):
    return c.post("/signup", data={"name": "New Person", "email": email, "password": password,
                                   "confirm": confirm or password, **extra})


# -- flows ---------------------------------------------------------------------------
def test_signup_login_logout_flow(guest, app):
    r = signup(guest)
    assert r.status_code == 302 and r.headers["Location"] == "/my-health"
    page = guest.get("/my-health").get_data(as_text=True)
    assert "Complete your profile" in page and "New Person" in page
    assert guest.get("/api/profile").status_code == 200

    assert guest.get("/logout").status_code == 405                  # POST only
    assert guest.post("/logout").status_code == 302
    assert guest.get("/api/profile").status_code == 401

    r = guest.login("new@example.com", "goodpass1")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert guest.get("/api/profile").status_code == 200
    u = user_row(app, "new@example.com")
    assert u.role == "user" and u.last_login_at is not None


def test_signup_lowercases_email_and_rejects_duplicates(guest, make_user):
    make_user("taken@example.com")
    r = signup(guest, email="Taken@Example.com")
    assert r.status_code == 200 and "already exists" in r.get_data(as_text=True)


@pytest.mark.parametrize("pw", ["short1", "lettersonly", "12345678", ""])
def test_weak_password_rejected(guest, app, pw):
    r = signup(guest, email="weak@example.com", password=pw)
    assert r.status_code == 200
    assert user_row(app, "weak@example.com") is None


def test_password_confirmation_must_match(guest, app):
    signup(guest, email="mm@example.com", password="goodpass1", confirm="goodpass2")
    assert user_row(app, "mm@example.com") is None


def test_role_cannot_be_chosen_at_signup(guest, app):
    signup(guest, email="sneaky@example.com", role="admin")
    assert user_row(app, "sneaky@example.com").role == "user"


def test_password_is_stored_hashed(guest, app):
    signup(guest, email="hash@example.com", password="goodpass1")
    h = user_row(app, "hash@example.com").password_hash
    assert h != "goodpass1" and "goodpass1" not in h
    assert h.startswith(("scrypt:", "pbkdf2:")) and check_password_hash(h, "goodpass1")
    raw = sqlite3.connect(app.config["SQLALCHEMY_DATABASE_URI"].removeprefix("sqlite:///"))
    assert all("goodpass1" not in str(row) for row in raw.execute("SELECT * FROM users"))


def test_login_error_is_generic(guest, make_user):
    make_user("real@example.com")
    wrong_pw = guest.login("real@example.com", "nottheone1").get_data(as_text=True)
    no_user = guest.login("ghost@example.com", "nottheone1").get_data(as_text=True)
    for page in (wrong_pw, no_user):
        assert "Invalid email or password." in page
    assert "no account" not in no_user.lower() and "wrong password" not in wrong_pw.lower()


def test_lockout_after_5_failures(guest, app, make_user):
    make_user("lock@example.com")
    for _ in range(5):
        assert guest.login("lock@example.com", "wrongpass1").status_code == 200
    u = user_row(app, "lock@example.com")
    assert u.locked_until is not None and u.locked_until > utcnow() + timedelta(minutes=14)
    # the right password does not help while locked, and the message stays generic
    r = guest.login("lock@example.com")
    assert r.status_code == 200 and "Invalid email or password." in r.get_data(as_text=True)
    # after 15 minutes it works again
    from dashboard.extensions import db
    with app.app_context():
        row = User.query.filter_by(email="lock@example.com").one()
        row.locked_until = utcnow() - timedelta(seconds=1)
        db.session.commit()
    assert guest.login("lock@example.com").status_code == 302


def test_deactivated_user_cannot_log_in(guest, make_user):
    make_user("off@example.com", is_active=False)
    r = guest.login("off@example.com")
    assert r.status_code == 200 and "Invalid email or password." in r.get_data(as_text=True)


# -- redirects -----------------------------------------------------------------------
@pytest.mark.parametrize("target", ["https://evil.com", "//evil.com", "/\\evil.com", "http:/evil.com",
                                    "javascript:alert(1)", "evil.com", "\x00/x"])
def test_open_redirect_blocked(guest, make_user, target):
    make_user("redir@example.com")
    r = guest.login("redir@example.com", next=target)
    assert r.status_code == 302 and r.headers["Location"] == "/"


def test_same_site_next_is_kept(guest, make_user):
    make_user("next@example.com")
    r = guest.login("next@example.com", next="/my-health?loc=Sylhet")
    assert r.headers["Location"] == "/my-health?loc=Sylhet"


def test_guest_page_redirects_to_login_with_next(guest):
    r = guest.get("/my-risk")
    assert r.status_code == 302 and "/login?next=/my-risk" in r.headers["Location"]


def test_safe_next_unit():
    assert safe_next("/a/b?c=1") == "/a/b?c=1"
    for bad in (None, "", "https://x.y", "//x.y", "/\\x.y", "x.y", " //x.y"):
        assert safe_next(bad) == "/"


# -- CSRF ------------------------------------------------------------------------------
def test_post_without_csrf_token_is_rejected(client, guest, make_user):
    client.csrf = False
    r = client.post("/api/profile", json={"age": 40})
    assert r.status_code == 400 and r.get_json()["error"] == "CSRF check failed"
    assert client.post("/api/settings", json={"units": "F"}).status_code == 400
    assert client.delete("/api/profile").status_code == 400
    assert client.post("/logout").status_code == 400
    make_user("csrf@example.com")
    guest.csrf = False
    assert guest.login("csrf@example.com").status_code == 400


def test_csrf_token_from_another_session_is_rejected(app, client):
    other = app.test_client()
    token = other.token()
    client.csrf = False
    r = client.post("/api/profile", json={"age": 40}, headers={"X-CSRFToken": token})
    assert r.status_code == 400


# -- sessions, cookies, headers ------------------------------------------------------------
def test_cookie_flags_and_remember_me(guest, make_user):
    make_user("rem@example.com")
    r = guest.login("rem@example.com", remember="y")
    cookies = r.headers.getlist("Set-Cookie")
    session = next(c for c in cookies if c.startswith("session="))
    remember = next(c for c in cookies if c.startswith("remember_token="))
    for c in (session, remember):
        assert "HttpOnly" in c and "SameSite=Lax" in c
    m = re.search(r"Max-Age=(\d+)", remember) or re.search(r"Expires=([^;]+)", remember)
    assert m and (not m.group(1).isdigit() or int(m.group(1)) == 14 * 24 * 3600)


def test_security_headers(guest):
    r = guest.get("/")
    h = r.headers
    assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"
    assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"
    csp = h["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in csp and "'nonce-" in csp and "unsafe-eval" not in csp
    page = r.get_data(as_text=True)
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    assert all(f'nonce="{nonce}"' in tag for tag in re.findall(r"<script(?! src)[^>]*>", page))


def test_password_change_signs_out_other_sessions(app, client):
    other = app.test_client()
    other.login("user@example.com")
    assert other.get("/api/profile").status_code == 200
    r = client.post("/account/password", data={"current": PASSWORD, "password": "newpass99",
                                                "confirm": "newpass99"})
    assert r.status_code == 302
    assert client.get("/api/profile").status_code == 200           # this session continues
    assert other.get("/api/profile").status_code == 401            # the other one ended


def test_password_change_needs_current_password(client):
    r = client.post("/account/password", data={"current": "wrongpass1", "password": "newpass99",
                                                "confirm": "newpass99"})
    assert r.status_code == 400 and "incorrect" in r.get_data(as_text=True)


def test_account_rename_and_delete(app, client, db_profile):
    client.post("/account/name", data={"name": "Renamed"})
    assert "Renamed" in client.get("/account").get_data(as_text=True)
    client.post("/api/profile", json={"asthma": True, "consent": True})
    client.post("/api/settings", json={"units": "F"})
    assert client.post("/account/delete", data={"password": "wrongpass1"}).status_code == 400
    r = client.post("/account/delete", data={"password": PASSWORD})
    assert r.status_code == 302 and client.get("/api/profile").status_code == 401
    raw = sqlite3.connect(app.config["SQLALCHEMY_DATABASE_URI"].removeprefix("sqlite:///"))
    for table in ("users", "profiles", "settings"):
        assert raw.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_forced_password_change_after_admin_reset(app, admin_client, make_user):
    uid = make_user("temp@example.com")
    temp = admin_client.post(f"/api/admin/users/{uid}/reset-password", json={}).get_json()["temp_password"]
    c = app.test_client()
    r = c.login("temp@example.com", temp)
    assert r.status_code == 302 and r.headers["Location"] == "/change-password"
    assert c.get("/").headers["Location"] == "/change-password"
    assert c.get("/api/dashboard").status_code == 403
    assert c.post("/change-password", data={"password": "weak", "confirm": "weak"}).status_code == 200
    assert c.post("/change-password", data={"password": "mynewpass7", "confirm": "mynewpass7"}).status_code == 302
    assert c.get("/api/dashboard").status_code == 200
    assert not user_row(app, "temp@example.com").must_change_password


# -- rate limit ---------------------------------------------------------------------------
def test_login_rate_limit(svc, tmp_path):
    from app import create_app
    from conftest import CSRFClient
    from dashboard.extensions import limiter
    application = create_app(svc, config={"TESTING": True, "SECRET_KEY": "k", "RATELIMIT_ENABLED": True,
                                          "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'rl.db'}"})
    application.test_client_class = CSRFClient
    limiter.reset()
    c = application.test_client()
    codes = [c.login("x@example.com", "whatever1").status_code for _ in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429
    limiter.reset()


# -- audit -------------------------------------------------------------------------------
def test_login_events_are_audited_without_passwords(app, guest, make_user):
    make_user("aud@example.com")
    guest.login("aud@example.com", "badguess77")
    guest.login("aud@example.com")
    guest.post("/logout")
    with app.app_context():
        rows = AuditLog.query.all()
        actions = [r.action for r in rows]
        text = " ".join(f"{r.action} {r.target}" for r in rows)
    assert {"login_fail", "login_success", "logout"} <= set(actions)
    assert "badguess77" not in text and PASSWORD not in text


# -- CLI ------------------------------------------------------------------------------------
def test_cli_init_db_and_create_admin(app, monkeypatch):
    runner = app.test_cli_runner()
    assert "Database ready" in runner.invoke(args=["init-db"]).output
    answers = iter(["adminpass1", "adminpass1"])
    monkeypatch.setattr("dashboard.cli.getpass", lambda _prompt: next(answers))
    r = runner.invoke(args=["create-admin", "--email", "Boss@Example.com", "--name", "Boss"])
    assert r.exit_code == 0, r.output
    u = user_row(app, "boss@example.com")
    assert u.role == "admin" and check_password_hash(u.password_hash, "adminpass1")
    assert "adminpass1" not in r.output

    weak = iter(["short", "short"])
    monkeypatch.setattr("dashboard.cli.getpass", lambda _prompt: next(weak))
    r = runner.invoke(args=["create-admin", "--email", "b2@example.com", "--name", "B"])
    assert r.exit_code != 0 and user_row(app, "b2@example.com") is None


def test_cli_create_admin_imports_legacy_profile(app, monkeypatch, tmp_path, db_profile):
    legacy = tmp_path / "legacy.db"
    con = sqlite3.connect(legacy)
    for t in ("profile", "settings"):
        con.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL)")
    con.execute("INSERT INTO profile VALUES (1, ?)", ('{"age": 61, "asthma": true, "consent": true}',))
    con.execute("INSERT INTO settings VALUES (1, ?)", ('{"units": "F"}',))
    con.execute("CREATE TABLE risk_history (x)")
    con.commit()
    con.close()
    monkeypatch.setattr("dashboard.cli.LEGACY_DB", legacy)
    answers = iter(["adminpass1", "adminpass1"])
    monkeypatch.setattr("dashboard.cli.getpass", lambda _prompt: next(answers))
    r = app.test_cli_runner().invoke(args=["create-admin", "--email", "own@example.com", "--name", "Owner",
                                           "--import-legacy"])
    assert r.exit_code == 0 and "Imported legacy profile, settings" in r.output
    p = db_profile("own@example.com")
    assert p["age"] == 61 and p["asthma"] is True
    tables = {x[0] for x in sqlite3.connect(legacy).execute("SELECT name FROM sqlite_master")}
    assert tables == {"risk_history"}                                # old tables dropped, history kept


def test_cli_seed_demo(app, db_profile):
    r = app.test_cli_runner().invoke(args=["seed-demo"])
    assert r.exit_code == 0
    creds = dict(re.findall(r"(\S+@example\.com)\s+(\S+)", r.output))
    assert set(creds) == {"demo-admin@example.com", "rahim@example.com", "karima@example.com"}
    assert user_row(app, "demo-admin@example.com").role == "admin"
    assert db_profile("karima@example.com")["asthma"] is True
    c = app.test_client()
    assert c.login("karima@example.com", creds["karima@example.com"]).status_code == 302
