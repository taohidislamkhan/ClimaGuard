"""Admin panel (/admin/*) and its JSON API (/api/admin/*).

Privacy rule: admins see account metadata only. Nothing here reads a profile
or settings row, so health data cannot leak into an admin response.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta

from flask import Blueprint, Response, abort, current_app, jsonify, request
from flask_login import current_user
from sqlalchemy import or_

from . import personal_rules
from .auth import hash_password, is_last_active_admin
from .extensions import db
from .inference import MODEL_DIR, MODEL_NAMES, TASKS
from .models import ROLES, AuditLog, User, new_session_token, utcnow
from .pages import model_health
from .security import audit, role_required, temp_password
from .views import ADMIN_PAGES, render_page
from src.utils import paths

bp = Blueprint("admin", __name__)
admin_only = role_required("admin")

AUDIT_ACTIONS = ["login_success", "login_fail", "signup", "logout", "password_change",
                 "role_change", "deactivate", "activate", "password_reset", "unlock",
                 "profile_delete", "account_delete", "recalc", "cache_clear"]
PER_PAGE = 10


def _page(key: str, template: str, **ctx):
    path, title, sub = ADMIN_PAGES[key]
    return render_page(template, key, f"Admin · {title}", sub, **ctx)


def _json_body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400, description="Expected a JSON object")
    return body


def _int_arg(name: str, default: int) -> int:
    try:
        return max(1, int(request.args.get(name, default)))
    except ValueError:
        abort(400, description=f"{name} must be an integer")


def _mtime(path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return None


# -- pages ----------------------------------------------------------------------------
@bp.get("/admin")
@admin_only
def overview():
    return _page("admin_overview", "admin/overview.html", stats=stats(), system=system_status())


@bp.get("/admin/users")
@admin_only
def users():
    return _page("admin_users", "admin/users.html")


@bp.get("/admin/model")
@admin_only
def model():
    def load(p):
        return json.loads(p.read_text()) if p.exists() else None
    return _page("admin_model", "admin/model.html", classifier=load(paths.CLASSIFIER_METRICS),
                 regressors=load(paths.REGRESSOR_METRICS), drift=model_health(),
                 model_names=MODEL_NAMES, system=system_status())


@bp.get("/admin/rules")
@admin_only
def rules():
    return _page("admin_rules", "admin/rules.html", rules=personal_rules.rule_table(),
                 matrix=personal_rules.MATRIX, overrides=personal_rules.CFG["overrides"],
                 sources=list(personal_rules.SOURCES.values()))


@bp.get("/admin/audit")
@admin_only
def audit_page():
    return _page("admin_audit", "admin/audit.html", actions=AUDIT_ACTIONS)


# -- overview data -----------------------------------------------------------------------
def stats() -> dict:
    now = utcnow()
    week = now - timedelta(days=7)
    return {"total": User.query.count(),
            "active_7d": User.query.filter(User.last_login_at >= week).count(),
            "new_7d": User.query.filter(User.created_at >= week).count(),
            "locked": User.query.filter(User.locked_until > now).count()}


def system_status() -> dict:
    svc = current_app.config["SERVICE"]
    models = [{"task": t, "model": MODEL_NAMES.get(t), "file": fn, "updated": _mtime(MODEL_DIR / fn)}
              for t, fn in TASKS.items()]
    drift = model_health()
    return {"models": models,
            "last_repro": _mtime(paths.ROOT / "dvc.lock"),
            "metrics_updated": _mtime(paths.CLASSIFIER_METRICS),
            "drift": None if drift is None else {
                "retrain_recommended": drift["retrain_recommended"], "reasons": drift["reasons"],
                "n_drifted": drift["data"]["n_drifted"], "n_features": drift["data"]["n_features"],
                "concept_flagged": drift["concept"]["n_flagged"]},
            "run": svc.run_status(), "ttl_minutes": svc.ttl_minutes}


@bp.get("/api/admin/overview")
@admin_only
def api_overview():
    return jsonify(stats=stats(), system=system_status())


# -- users ---------------------------------------------------------------------------------
@bp.get("/api/admin/users")
@admin_only
def api_users():
    q = (request.args.get("q") or "").strip()[:100]
    page = _int_arg("page", 1)
    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(User.name.ilike(like), User.email.ilike(like)))
    total = query.count()
    rows = query.order_by(User.created_at.desc(), User.id.desc()) \
                .offset((page - 1) * PER_PAGE).limit(PER_PAGE).all()
    return jsonify(items=[{**u.account_dict(), "is_self": u.id == current_user.id} for u in rows],
                   total=total, page=page, pages=max(1, -(-total // PER_PAGE)), per_page=PER_PAGE)


def _target(uid: int) -> User:
    user = db.session.get(User, uid)
    if user is None:
        abort(404, description="No such user")
    return user


@bp.post("/api/admin/users/<int:uid>/role")
@admin_only
def api_role(uid: int):
    user, role = _target(uid), _json_body().get("role")
    if role not in ROLES:
        abort(400, description=f"role must be one of {list(ROLES)}")
    if role == user.role:
        return jsonify(user=user.account_dict())
    if user.id == current_user.id:
        abort(400, description="You cannot change your own role.")
    if role != "admin" and is_last_active_admin(user):
        abort(400, description="At least one active admin must remain.")
    old, user.role = user.role, role
    db.session.commit()
    audit("role_change", f"user:{user.id} {old}->{role}")
    return jsonify(user=user.account_dict())


@bp.post("/api/admin/users/<int:uid>/status")
@admin_only
def api_status(uid: int):
    user, active = _target(uid), _json_body().get("active")
    if not isinstance(active, bool):
        abort(400, description="active must be true or false")
    if active == user.is_active:
        return jsonify(user=user.account_dict())
    if user.id == current_user.id:
        abort(400, description="You cannot deactivate yourself.")
    if not active and is_last_active_admin(user):
        abort(400, description="At least one active admin must remain.")
    user.is_active = active
    if not active:
        user.session_token = new_session_token()       # signs the user out everywhere
    db.session.commit()
    audit("activate" if active else "deactivate", f"user:{user.id}")
    return jsonify(user=user.account_dict())


@bp.post("/api/admin/users/<int:uid>/reset-password")
@admin_only
def api_reset_password(uid: int):
    user = _target(uid)
    if user.id == current_user.id:
        abort(400, description="Use the Account page to change your own password.")
    pw = temp_password()
    user.password_hash = hash_password(pw)
    user.must_change_password = True
    user.failed_logins, user.locked_until = 0, None
    user.session_token = new_session_token()
    db.session.commit()
    audit("password_reset", f"user:{user.id}")        # the temporary password is never logged
    return jsonify(user=user.account_dict(), temp_password=pw)


@bp.post("/api/admin/users/<int:uid>/unlock")
@admin_only
def api_unlock(uid: int):
    user = _target(uid)
    user.failed_logins, user.locked_until = 0, None
    db.session.commit()
    audit("unlock", f"user:{user.id}")
    return jsonify(user=user.account_dict())


# -- model & data actions ---------------------------------------------------------------
@bp.post("/api/admin/recalculate")
@admin_only
def api_recalculate():
    """Re-run the models for all eight divisions (the existing recalc logic) and
    store today's snapshot for each. Never runs ``dvc repro``."""
    svc = current_app.config["SERVICE"]
    from .weather_client import DIVISIONS
    svc.model_run(force=True)
    for loc in DIVISIONS:
        svc.dashboard(loc)
    audit("recalc", "all divisions")
    return jsonify(ok=True, divisions=len(DIVISIONS), run=svc.run_status())


@bp.post("/api/admin/clear-cache")
@admin_only
def api_clear_cache():
    current_app.config["SERVICE"].clear_cache()
    audit("cache_clear", "weather + model run")
    return jsonify(ok=True)


# -- audit log ------------------------------------------------------------------------
def _audit_query():
    query = AuditLog.query
    action = request.args.get("action") or ""
    if action:
        query = query.filter(AuditLog.action == action)
    who = (request.args.get("user") or "").strip()[:100]
    if who:
        like = f"%{who}%"
        ids = [u.id for u in User.query.filter(or_(User.email.ilike(like), User.name.ilike(like)))]
        query = query.filter(or_(AuditLog.actor_user_id.in_(ids), AuditLog.target.ilike(like)))
    for arg, op in (("from", "ge"), ("to", "lt")):
        v = request.args.get(arg)
        if v:
            try:
                d = datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                abort(400, description=f"{arg} must be YYYY-MM-DD")
            query = query.filter(AuditLog.timestamp >= d if op == "ge"
                                 else AuditLog.timestamp < d + timedelta(days=1))
    return query.order_by(AuditLog.timestamp.desc(), AuditLog.id.desc())


def _with_actor(rows) -> list[dict]:
    ids = {r.actor_user_id for r in rows if r.actor_user_id}
    emails = {u.id: u.email for u in User.query.filter(User.id.in_(ids))} if ids else {}
    return [{**r.as_dict(), "actor": emails.get(r.actor_user_id)} for r in rows]


@bp.get("/api/admin/audit")
@admin_only
def api_audit():
    page = _int_arg("page", 1)
    query = _audit_query()
    total = query.count()
    rows = query.offset((page - 1) * 20).limit(20).all()
    return jsonify(items=_with_actor(rows), total=total, page=page, pages=max(1, -(-total // 20)))


def _csv_cell(v) -> str:
    """Neutralise spreadsheet formulas (CSV injection)."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


@bp.get("/api/admin/audit.csv")
@admin_only
def api_audit_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    cols = ["id", "timestamp", "action", "actor_user_id", "actor", "target", "ip"]
    w.writerow(cols)
    for r in _with_actor(_audit_query().limit(10000).all()):
        w.writerow([_csv_cell(r[c]) for c in cols])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=audit_log.csv"})
