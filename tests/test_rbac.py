"""Role-based access: guest / user / admin, object-level checks and the admin privacy rule."""

import re

import pytest

from dashboard.auth import is_last_active_admin
from dashboard.models import AuditLog, User

PERSONAL_API = [
    ("get", "/api/profile"), ("post", "/api/profile"), ("delete", "/api/profile"),
    ("get", "/api/advisory/personal"), ("post", "/api/profile/preview"),
    ("get", "/api/settings"), ("post", "/api/settings"),
    ("get", "/api/disease/heat"), ("post", "/api/recalculate"),
]
ADMIN_PAGES = ["/admin", "/admin/users", "/admin/model", "/admin/rules", "/admin/audit"]
ADMIN_API = [("get", "/api/admin/overview"), ("get", "/api/admin/users"), ("get", "/api/admin/audit"),
             ("get", "/api/admin/audit.csv"), ("post", "/api/admin/clear-cache"),
             ("post", "/api/admin/users/1/unlock")]
PUBLIC = ["/", "/environment", "/risk-map", "/risk-history", "/how-it-works", "/login", "/signup"]
HEALTH_WORDS = ["asthma", "diabetes", "pregnan", "cardiovascular", "immunity", "bmi", "consent",
                "outdoor_hours", "water_source", "smoking"]


def call(c, method, url):
    return getattr(c, method)(url, json={})


# -- guests -------------------------------------------------------------------------------
@pytest.mark.parametrize("method,url", PERSONAL_API)
def test_guest_gets_json_401_on_personal_apis(guest, method, url):
    r = call(guest, method, url)
    assert r.status_code == 401 and r.is_json and r.get_json()["error"] == "Unauthorized"


@pytest.mark.parametrize("url", ["/my-risk", "/my-health", "/settings", "/account"] + ADMIN_PAGES)
def test_guest_pages_redirect_to_login(guest, url):
    r = guest.get(url)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


@pytest.mark.parametrize("method,url", ADMIN_API)
def test_guest_gets_401_on_admin_api(guest, method, url):
    assert call(guest, method, url).status_code == 401


@pytest.mark.parametrize("url", PUBLIC)
def test_public_pages_open_to_guests(guest, url):
    assert guest.get(url).status_code == 200


def test_guest_dashboard_is_regional_only(guest):
    d = guest.get("/api/dashboard").get_json()
    assert d["profile"] is None
    assert all(x["adjustment"]["points"] == 0 for x in d["diseases"])
    page = guest.get("/").get_data(as_text=True)
    assert "Log in for personalized advice" in page and 'id="recalc"' not in page
    assert '"user": null' in page


# -- users ----------------------------------------------------------------------------------
@pytest.mark.parametrize("url", ADMIN_PAGES)
def test_user_gets_403_page_on_admin(client, url):
    r = client.get(url)
    assert r.status_code == 403 and "You don't have access to this page" in r.get_data(as_text=True)


@pytest.mark.parametrize("method,url", ADMIN_API)
def test_user_gets_json_403_on_admin_api(client, method, url):
    r = call(client, method, url)
    assert r.status_code == 403 and r.is_json


def test_user_sidebar_has_no_admin_section(client):
    page = client.get("/").get_data(as_text=True)
    assert 'href="/admin"' not in page and "Log out" in page


# -- admins ----------------------------------------------------------------------------------
@pytest.mark.parametrize("url", ADMIN_PAGES)
def test_admin_gets_admin_pages(admin_client, url):
    r = admin_client.get(url)
    assert r.status_code == 200 and 'href="/admin/users"' in r.get_data(as_text=True)


@pytest.mark.parametrize("method,url", ADMIN_API)
def test_admin_gets_200_on_admin_api(admin_client, method, url):
    assert call(admin_client, method, url).status_code == 200


def test_admin_recalculates_all_divisions(admin_client):
    r = admin_client.post("/api/admin/recalculate", json={})
    assert r.status_code == 200 and r.get_json()["divisions"] == 8


def test_admin_personal_pages_show_own_data_only(admin_client):
    assert admin_client.get("/api/profile").get_json()["saved"] is False
    assert admin_client.get("/my-health").status_code == 200


# -- object-level checks --------------------------------------------------------------------
def test_user_a_cannot_read_or_modify_user_b(app, make_user, db_profile):
    a_id = make_user("a@example.com")
    b_id = make_user("b@example.com")
    a, b = app.test_client(), app.test_client()
    a.login("a@example.com")
    b.login("b@example.com")
    assert a.post("/api/profile", json={"age": 66, "asthma": True, "consent": True}).status_code == 200

    # B sees only B's own (default) profile, whatever ids are passed
    for url in ("/api/profile", f"/api/profile?user_id={a_id}", f"/api/profile?id={a_id}"):
        p = b.get(url).get_json()
        assert p["saved"] is False and p["asthma"] is False and p["name"] == "B"
    # B's writes land on B's row, even with A's id in the body or the query
    b.post(f"/api/profile?user_id={a_id}", json={"user_id": a_id, "id": a_id, "age": 30})
    b.delete(f"/api/profile?user_id={a_id}")
    b.post("/api/settings", json={"user_id": a_id, "units": "F"})
    assert db_profile("a@example.com")["age"] == 66 and db_profile("a@example.com")["asthma"] is True
    assert a.get("/api/settings").get_json()["units"] == "C"
    # personal APIs for B never reflect A's conditions
    adv = b.get("/api/advisory/personal").get_json()
    assert adv["personalized"] is False
    assert b_id != a_id


# -- privacy: admins never see health data --------------------------------------------------
def test_admin_responses_contain_no_health_data(app, admin_client, make_user):
    make_user("sick@example.com", name="Patient")
    u = app.test_client()
    u.login("sick@example.com")
    u.post("/api/profile", json={"asthma": True, "diabetes": True, "pregnancy": True, "consent": True,
                                 "bmi": 31.7, "outdoor_hours": 9, "water_source": "tube_well"})
    users = admin_client.get("/api/admin/users?q=sick").get_json()["items"]
    assert len(users) == 1
    assert set(users[0]) == {"id", "name", "email", "role", "is_active", "locked", "must_change_password",
                             "created_at", "last_login_at", "is_self"}
    responses = [admin_client.get(url) for url in
                 ["/api/admin/users", "/api/admin/overview", "/api/admin/audit", "/api/admin/audit.csv",
                  "/admin", "/admin/users", "/admin/audit"]]
    for r in responses:
        body = r.get_data(as_text=True).lower()
        assert r.status_code == 200
        for w in HEALTH_WORDS + ["31.7", "tube_well"]:
            assert not re.search(rf"{w}", body), (r.request.path, w)


# -- admin actions ------------------------------------------------------------------------------
def admin_id(app):
    with app.app_context():
        return User.query.filter_by(email="admin@example.com").one().id


def test_last_admin_cannot_be_demoted_or_deactivated(app, admin_client):
    me = admin_id(app)
    r = admin_client.post(f"/api/admin/users/{me}/role", json={"role": "user"})
    assert r.status_code == 400
    assert admin_client.post(f"/api/admin/users/{me}/status", json={"active": False}).status_code == 400
    r = admin_client.post("/account/delete", data={"password": "secret123"})
    assert r.status_code == 400 and "only active admin" in r.get_data(as_text=True)
    with app.app_context():
        u = User.query.get(me)
        assert u.role == "admin" and u.is_active and is_last_active_admin(u)


def test_admin_can_demote_another_admin_but_one_always_remains(app, admin_client, make_user):
    other = make_user("admin2@example.com", role="admin")
    assert admin_client.post(f"/api/admin/users/{other}/role", json={"role": "user"}).status_code == 200
    with app.app_context():
        assert User.query.filter_by(role="admin", is_active=True).count() == 1


def test_admin_actions_are_audited_without_secrets(app, admin_client, make_user):
    uid = make_user("target@example.com")
    t = app.test_client()
    t.login("target@example.com")
    t.post("/api/profile", json={"asthma": True, "consent": True})
    steps = [("role", {"role": "admin"}), ("role", {"role": "user"}), ("status", {"active": False}),
             ("status", {"active": True}), ("unlock", {}), ("reset-password", {})]
    temp = None
    for path, body in steps:
        r = admin_client.post(f"/api/admin/users/{uid}/{path}", json=body)
        assert r.status_code == 200, (path, r.get_json())
        temp = r.get_json().get("temp_password") or temp
    admin_client.post("/api/admin/clear-cache", json={})
    with app.app_context():
        rows = AuditLog.query.filter(AuditLog.actor_user_id == admin_id(app)).all()
        actions = {r.action for r in rows}
        dump = " ".join(f"{r.action} {r.target} {r.ip}" for r in AuditLog.query.all()).lower()
    assert {"role_change", "deactivate", "activate", "unlock", "password_reset", "cache_clear"} <= actions
    assert all(r.ip for r in rows)
    assert temp and temp.lower() not in dump and "secret123" not in dump
    for w in HEALTH_WORDS:
        assert w not in dump
    csv = admin_client.get("/api/admin/audit.csv").get_data(as_text=True)
    assert csv.startswith("id,timestamp,action") and temp not in csv


def test_deactivation_ends_sessions(app, admin_client, make_user):
    uid = make_user("bye@example.com")
    c = app.test_client()
    c.login("bye@example.com")
    assert c.get("/api/profile").status_code == 200
    admin_client.post(f"/api/admin/users/{uid}/status", json={"active": False})
    assert c.get("/api/profile").status_code == 401


def test_user_search_pagination_and_audit_filters(app, admin_client, make_user):
    for i in range(12):
        make_user(f"bulk{i:02d}@example.com")
    r = admin_client.get("/api/admin/users?page=2").get_json()
    assert r["total"] == 13 and r["pages"] == 2 and len(r["items"]) == 3
    assert admin_client.get("/api/admin/users?q=bulk0").get_json()["total"] == 10
    assert admin_client.get("/api/admin/users?page=x").status_code == 400
    a = admin_client.get("/api/admin/audit?action=login_success").get_json()
    assert a["total"] >= 1 and all(x["action"] == "login_success" for x in a["items"])
    assert admin_client.get("/api/admin/audit?user=admin@example.com").get_json()["total"] >= 1
    assert admin_client.get("/api/admin/audit?from=2000-01-01&to=2000-01-02").get_json()["total"] == 0
    assert admin_client.get("/api/admin/audit?from=yesterday").status_code == 400


def test_csv_export_neutralises_formulas():
    from dashboard.admin import _csv_cell
    assert _csv_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert _csv_cell("+1") == "'+1" and _csv_cell("@x") == "'@x" and _csv_cell("-2") == "'-2"
    assert _csv_cell("user:5") == "user:5" and _csv_cell(None) == ""
