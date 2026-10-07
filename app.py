"""ClimaGuard — Flask dashboard.

Run::

    pip install -r requirements.txt
    dvc pull          # models, data and artifacts/ from the DVC remote (or: dvc repro)
    cp .env.example .env   # then set SECRET_KEY
    flask init-db && flask create-admin --email you@example.com --name You
    python app.py     # http://127.0.0.1:5000

The app only reads DVC pipeline outputs; it never trains. Accounts live in
``instance/app.db`` (git-ignored, not DVC-tracked).
"""

from __future__ import annotations

import os
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError
from werkzeug.exceptions import HTTPException

from dashboard import admin, auth, cli, storage
from dashboard.extensions import csrf, db, limiter, login_manager
from dashboard.models import User
from dashboard.pages import ArtifactMissing, PageData
from dashboard.security import audit, csp_nonce, login_required, set_security_headers
from dashboard.service import DashboardService
from dashboard.storage import ValidationError
from dashboard.views import PAGES, PERSONAL, render_page
from dashboard.weather_client import DIVISIONS

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def create_app(service: DashboardService | None = None, config: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    secure = _env_flag("SESSION_COOKIE_SECURE")
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY"),
        SQLALCHEMY_DATABASE_URI=os.environ.get(
            "DATABASE_URL", "sqlite:///" + os.path.join(app.instance_path, "app.db")),
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=secure,
        REMEMBER_COOKIE_DURATION=timedelta(days=14), REMEMBER_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_SAMESITE="Lax", REMEMBER_COOKIE_SECURE=secure,
        WTF_CSRF_HEADERS=["X-CSRFToken"], WTF_CSRF_TIME_LIMIT=None,
    )
    app.config.update(config or {})
    if not app.config["SECRET_KEY"]:
        raise RuntimeError("SECRET_KEY is not set. Copy .env.example to .env and set a random value "
                           "(python -c 'import secrets; print(secrets.token_hex(32))').")
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    login_manager.init_app(app)
    login_manager.session_protection = "strong"
    login_manager.login_view = "auth.login"
    login_manager.login_message = None
    app.register_blueprint(auth.bp)
    app.register_blueprint(admin.bp)
    cli.register(app)
    with app.app_context():
        db.create_all()

    svc = service or DashboardService()
    data = PageData(svc)
    app.config["SERVICE"] = svc
    app.config["PAGES_DATA"] = data

    # -- sessions ---------------------------------------------------------------
    @login_manager.user_loader
    def load_user(token: str):
        # The cookie holds the rotating session token, not the user id, so a
        # password change, reset or deactivation ends every old session.
        user = User.query.filter_by(session_token=token).first()
        return user if user is not None and user.is_active else None

    @login_manager.unauthorized_handler
    def unauthorized():
        if request.path.startswith("/api/"):
            return jsonify(error="Unauthorized", detail="Log in to use this endpoint."), 401
        return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))

    @app.before_request
    def force_password_change():
        if (current_user.is_authenticated and current_user.must_change_password
                and request.endpoint not in ("auth.change_password_forced", "auth.logout", "static")):
            if request.path.startswith("/api/"):
                return jsonify(error="Forbidden", detail="Change your temporary password first."), 403
            return redirect(url_for("auth.change_password_forced"))

    @app.context_processor
    def inject():
        return {"csp_nonce": csp_nonce}

    app.after_request(set_security_headers)

    # -- pages -----------------------------------------------------------------
    def page(key: str):
        if key in PERSONAL and not current_user.is_authenticated:
            return login_manager.unauthorized()
        return render_page(f"{key}.html", key, PAGES[key][1], PAGES[key][2])

    for key in PAGES:
        app.add_url_rule(PAGES[key][0], key, (lambda k=key: page(k)))

    # -- errors: JSON for the API, HTML otherwise --------------------------------
    @app.errorhandler(CSRFError)
    def csrf_error(e: CSRFError):
        if request.path.startswith("/api/"):
            return jsonify(error="CSRF check failed", detail=e.description), 400
        return render_template("errors/error.html", code=400, title="Your session expired",
                               message="The form was open too long or came from another site. "
                                       "Go back, reload the page and try again.",
                               settings={"theme": "light"}), 400

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify(error=e.name, detail=e.description), e.code
        if e.code == 403:
            return render_page("errors/403.html", "", "Access denied", "This area is for admins only."), 403
        if e.code == 429:
            return render_template("errors/error.html", code=429, title="Too many attempts",
                                   message="Please wait a minute and try again.",
                                   settings={"theme": "light"}), 429
        return e

    @app.errorhandler(ArtifactMissing)
    def artifact_missing(e):
        return jsonify(error="Artifact missing", detail=str(e)), 503

    @app.errorhandler(ValidationError)
    def invalid(e):
        return jsonify(error="Invalid input", detail=str(e)), 400

    # -- request helpers ------------------------------------------------------------
    def _user():
        return current_user._get_current_object() if current_user.is_authenticated else None

    def _location() -> str:
        body = request.get_json(silent=True) if request.is_json else None
        body = body if isinstance(body, dict) else {}
        loc = (request.args.get("loc") or request.args.get("location")
               or body.get("loc") or body.get("location")
               or storage.settings_for(_user())["default_location"])
        if loc not in DIVISIONS:
            abort(400, description=f"Unknown location {loc!r}; use one of {list(DIVISIONS)}")
        return loc

    def _json_body() -> dict:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            abort(400, description="Expected a JSON object")
        return body

    def _week() -> int | None:
        w = request.args.get("week")
        if w in (None, ""):
            return None
        try:
            return int(w)
        except ValueError:
            abort(400, description="week must be an integer index")

    def _or_404(fn, *args):
        try:
            return fn(*args)
        except KeyError as e:
            abort(404, description=f"Unknown value: {e}")

    # -- health check: no model, data or network access ------------------------------
    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    # -- dashboard ------------------------------------------------------------------
    # Personal data: every lookup goes through the signed-in user; no endpoint
    # takes a user id.
    def _my_profile() -> dict | None:
        return storage.profile_for(_user())

    @app.get("/api/dashboard")
    def api_dashboard():
        return jsonify(svc.dashboard(_location(), profile=_my_profile()))

    @app.post("/api/recalculate")
    @login_required
    @limiter.limit("6 per minute")
    def api_recalculate():
        loc = _location()
        out = svc.dashboard(loc, force=True, profile=_my_profile())
        audit("recalc", f"division:{loc}")
        return jsonify(out)

    # -- pages ------------------------------------------------------------------------
    @app.get("/api/disease/<name>")
    @login_required
    def api_disease(name: str):
        return jsonify(_or_404(data.disease, name, _location(), _my_profile()))

    @app.get("/api/environment")
    def api_environment():
        return jsonify(data.environment(_location()))

    @app.get("/api/forecast")
    def api_forecast():
        try:
            return jsonify(data.forecast(_location()))
        except (OSError, ValueError) as e:        # requests errors subclass OSError
            abort(502, description=f"Open-Meteo forecast unavailable: {e}")

    @app.get("/api/map")
    def api_map():
        return jsonify(_or_404(data.map, request.args.get("view", "division"),
                               request.args.get("layer", "overall"), _week()))

    @app.get("/api/map/detail")
    def api_map_detail():
        return jsonify(_or_404(data.map_detail, request.args.get("view", "division"),
                               request.args.get("id", ""), _week()))

    @app.get("/api/history")
    def api_history():
        return jsonify(_or_404(data.history, _location(), request.args.get("range", "12w")))

    @app.get("/api/history.csv")
    def api_history_csv():
        loc = _location()
        svc.ensure_backfill(loc)
        return Response(data.snapshots_csv(loc), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=risk_snapshots_{loc}.csv"})

    @app.get("/api/test-predictions")
    def api_test_predictions():
        return jsonify(_or_404(data.test_predictions, request.args.get("target", "respiratory")))

    @app.get("/api/seasonality")
    def api_seasonality():
        return jsonify(_or_404(data.seasonality, request.args.get("scope", "bgd")))

    @app.get("/api/methodology")
    def api_methodology():
        return jsonify(data.methodology())

    @app.route("/api/profile", methods=["GET", "POST", "DELETE"])
    @login_required
    def api_profile():
        user = _user()
        if request.method == "POST":
            return jsonify(storage.save_profile(user, _json_body()))
        if request.method == "DELETE":
            storage.delete_profile(user)
            audit("profile_delete", f"user:{user.id}")
            return jsonify(deleted=True, profile=storage.profile_or_default(user))
        return jsonify({**storage.profile_or_default(user), "saved": storage.has_profile(user)})

    def _lang() -> str:
        return "bn" if request.args.get("lang") == "bn" else "en"

    @app.get("/api/advisory/personal")
    @login_required
    def api_personal_advisory():
        return jsonify(_or_404(svc.personal_advisory, _location(), _my_profile(), _lang()))

    @app.post("/api/profile/preview")
    @login_required
    def api_profile_preview():
        return jsonify(data.profile_preview(_json_body(), _location(), _lang(),
                                            storage.profile_or_default(_user())))

    @app.route("/api/settings", methods=["GET", "POST"])
    @login_required
    def api_settings():
        if request.method == "POST":
            return jsonify(storage.save_settings(_user(), _json_body()))
        return jsonify(storage.settings_for(_user()))

    @app.get("/api/about")
    def api_about():
        return jsonify(data.about())

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
