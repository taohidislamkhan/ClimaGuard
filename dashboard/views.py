"""Page registry and the shared page renderer (sidebar, header, theme)."""

from __future__ import annotations

from flask import render_template
from flask_login import current_user

from . import storage
from .weather_client import DIVISIONS

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
# Pages with personal data: guests are sent to the login page.
PERSONAL = {"my_risk", "my_health", "settings"}

ADMIN_PAGES = {
    "admin_overview": ("/admin",          "Overview",       "Accounts and system status."),
    "admin_users":    ("/admin/users",    "Users",          "Account metadata only. Health data is visible to its owner alone."),
    "admin_model":    ("/admin/model",    "Model & Data",   "Read-only model metrics and the drift report."),
    "admin_rules":    ("/admin/rules",    "Advisory Rules", "The personal rule table and decision matrix, with sources."),
    "admin_audit":    ("/admin/audit",    "Audit Log",      "Security-relevant actions. No passwords or health data are logged."),
}


def nav_user() -> dict | None:
    if not current_user.is_authenticated:
        return None
    return {"name": current_user.name, "role": current_user.role}


def render_page(template: str, active: str, title: str, subtitle: str, **ctx):
    user = current_user if current_user.is_authenticated else None
    return render_template(template, active=active, pages=PAGES, admin_pages=ADMIN_PAGES,
                           personal_pages=PERSONAL, page_title=title, page_subtitle=subtitle,
                           settings=storage.settings_for(user), divisions=list(DIVISIONS),
                           user=nav_user(), **ctx)
