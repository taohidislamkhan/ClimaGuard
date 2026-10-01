"""ClimaGuard — Flask dashboard.

Run::

    pip install -r requirements.txt
    dvc pull          # models, data and artifacts/ from the DVC remote (or: dvc repro)
    python app.py     # http://127.0.0.1:5000

The app only reads DVC pipeline outputs; it never trains.
"""

from __future__ import annotations

from flask import Flask, Response, abort, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from dashboard.pages import ArtifactMissing, PageData
from dashboard.service import DashboardService
from dashboard.storage import ValidationError
from dashboard.weather_client import DIVISIONS

PAGES = {
    "dashboard":    ("/",              "Dashboard",     "Here's your environmental health overview for today."),
    "my_risk":      ("/my-risk",       "My Risk",       "A deep dive into each disease score: model, profile and drivers."),
    "environment":  ("/environment",   "Environment",   "Current conditions, the 7-day forecast and the next 72 hours of air quality."),
    "risk_map":     ("/risk-map",      "Risk Map",      "Risk by Bangladesh division (live) or by dataset country (test period)."),
    "risk_history": ("/risk-history",  "Risk History",  "How the risk scores evolve, and how well the models match reality."),
    "my_health":    ("/my-health",     "My Health",     "Your profile and exactly how it adjusts your scores."),
    "how":          ("/how-it-works",  "How It Works",  "Methodology, data, models, explainability and limitations."),
    "settings":     ("/settings",      "Settings",      "Units, theme, location, refresh interval and notifications."),
}


def create_app(service: DashboardService | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    svc = service or DashboardService()
    data = PageData(svc)
    app.config["SERVICE"] = svc
    app.config["PAGES_DATA"] = data

    # -- pages -----------------------------------------------------------------
    def page(key: str, template: str | None = None):
        return render_template(template or f"{key}.html", active=key, pages=PAGES,
                               page_title=PAGES[key][1], page_subtitle=PAGES[key][2],
                               settings=svc.app.settings(), divisions=list(DIVISIONS))

    for key in PAGES:
        app.add_url_rule(PAGES[key][0], key, (lambda k=key: page(k)))

    # -- errors: JSON for the API, HTML otherwise --------------------------------
    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify(error=e.name, detail=e.description), e.code
        return e

    @app.errorhandler(ArtifactMissing)
    def artifact_missing(e):
        return jsonify(error="Artifact missing", detail=str(e)), 503

    @app.errorhandler(ValidationError)
    def invalid(e):
        return jsonify(error="Invalid input", detail=str(e)), 400

    # -- request helpers ------------------------------------------------------------
    def _location() -> str:
        body = request.get_json(silent=True) if request.is_json else None
        body = body if isinstance(body, dict) else {}
        loc = (request.args.get("loc") or request.args.get("location")
               or body.get("loc") or body.get("location")
               or svc.app.settings()["default_location"])
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

    # -- dashboard ------------------------------------------------------------------
    @app.get("/api/dashboard")
    def api_dashboard():
        return jsonify(svc.dashboard(_location()))

    @app.post("/api/recalculate")
    def api_recalculate():
        return jsonify(svc.dashboard(_location(), force=True))

    # -- pages ------------------------------------------------------------------------
    @app.get("/api/disease/<name>")
    def api_disease(name: str):
        return jsonify(_or_404(data.disease, name, _location()))

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
    def api_profile():
        if request.method == "POST":
            return jsonify(svc.update_profile(_json_body()))
        if request.method == "DELETE":
            svc.app.delete_profile()
            return jsonify(deleted=True, profile=svc.profile)
        return jsonify({**svc.profile, "saved": svc.app.has_profile()})

    def _lang() -> str:
        return "bn" if request.args.get("lang") == "bn" else "en"

    @app.get("/api/advisory/personal")
    def api_personal_advisory():
        return jsonify(_or_404(svc.personal_advisory, _location(), None, _lang()))

    @app.post("/api/profile/preview")
    def api_profile_preview():
        return jsonify(data.profile_preview(_json_body(), _location(), _lang()))

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings():
        if request.method == "POST":
            s = svc.app.save_settings(_json_body())
            return jsonify(s)
        return jsonify(svc.app.settings())

    @app.get("/api/about")
    def api_about():
        return jsonify(data.about())

    return app


if __name__ == "__main__":
    import os
    import threading

    application = create_app()

    def warm():
        # Fetch live data + run the models, then build the SHAP explainers,
        # so the first page loads don't pay for either.
        application.config["SERVICE"].model_run()
        application.config["PAGES_DATA"].warm()

    threading.Thread(target=warm, daemon=True).start()
    application.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
