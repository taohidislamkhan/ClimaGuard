"""Sign up, log in, log out, account management and the forced password change."""

from __future__ import annotations

import secrets
from datetime import timedelta

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, limiter
from .forms import (ChangePasswordForm, DeleteAccountForm, ForcedPasswordForm, LoginForm, NameForm,
                    SignupForm)
from .models import User, new_session_token, utcnow
from .security import PASSWORD_RULE, audit, login_required, safe_next
from .views import render_page

bp = Blueprint("auth", __name__)

GENERIC_LOGIN_ERROR = "Invalid email or password."
MAX_FAILED = 5
LOCK_MINUTES = 15
_dummy_hash: str | None = None


def hash_password(pw: str) -> str:
    return generate_password_hash(pw)          # Werkzeug default: scrypt, random salt


def _dummy() -> str:
    """Hash checked when the email is unknown, so both paths take similar time."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_hex(16))
    return _dummy_hash


def active_admin_count() -> int:
    return User.query.filter_by(role="admin", is_active=True).count()


def is_last_active_admin(user: User) -> bool:
    return user.role == "admin" and user.is_active and active_admin_count() <= 1


def _start_session(user: User, remember: bool) -> None:
    """New session for a fresh login (drops anything from the old one)."""
    session.clear()
    login_user(user, remember=remember, duration=current_app.config["REMEMBER_COOKIE_DURATION"])


def _rotate_and_relogin(user: User) -> None:
    """After a password change: invalidate other sessions, keep this one."""
    remember = current_app.config.get("REMEMBER_COOKIE_NAME", "remember_token") in request.cookies
    user.session_token = new_session_token()
    db.session.commit()
    _start_session(user, remember)


# -- log in / sign up / log out ------------------------------------------------------
@bp.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if current_user.is_authenticated:
        return redirect(safe_next(request.args.get("next")))
    form = LoginForm()
    if request.method == "GET":
        form.next.data = safe_next(request.args.get("next"), "")
    error = None
    if form.validate_on_submit():
        email, now = form.email.data, utcnow()
        user = User.query.filter_by(email=email).first()
        ok = False
        if user is None or user.is_locked(now):
            check_password_hash(_dummy(), form.password.data)
        else:
            if user.locked_until:                          # lock expired: start counting again
                user.locked_until, user.failed_logins = None, 0
            if check_password_hash(user.password_hash, form.password.data):
                ok = user.is_active
            else:
                user.failed_logins += 1
                if user.failed_logins >= MAX_FAILED:
                    user.locked_until = now + timedelta(minutes=LOCK_MINUTES)
                    user.failed_logins = 0
            db.session.commit()
        if ok:
            user.failed_logins, user.locked_until, user.last_login_at = 0, None, now
            db.session.commit()
            _start_session(user, form.remember.data)
            audit("login_success", f"user:{user.id}", actor_id=user.id)
            if user.must_change_password:
                return redirect(url_for("auth.change_password_forced"))
            return redirect(safe_next(form.next.data))
        audit("login_fail", f"email:{email}", actor_id=user.id if user else None)
        error = GENERIC_LOGIN_ERROR
    elif request.method == "POST":
        error = GENERIC_LOGIN_ERROR
    return render_template("auth/login.html", form=form, error=error, user=None,
                           settings={"theme": "light"}, lock_minutes=LOCK_MINUTES, max_failed=MAX_FAILED)


@bp.route("/signup", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def signup():
    if current_user.is_authenticated:
        return redirect("/")
    form = SignupForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            form.email.errors.append("An account with this email already exists.")
        else:
            user = User(name=form.name.data, email=form.email.data,
                        password_hash=hash_password(form.password.data), role="user",
                        last_login_at=utcnow())
            db.session.add(user)
            db.session.commit()
            _start_session(user, remember=False)
            audit("signup", f"user:{user.id}", actor_id=user.id)
            flash("complete_profile", "welcome")
            return redirect("/my-health")
    return render_template("auth/signup.html", form=form, user=None, settings={"theme": "light"},
                           password_rule=PASSWORD_RULE)


@bp.post("/logout")
@login_required
def logout():
    audit("logout", f"user:{current_user.id}")
    logout_user()
    session.clear()
    return redirect("/")


# -- account ----------------------------------------------------------------------
def _account(name_form=None, pw_form=None, del_form=None, status=200):
    return render_page("account.html", "account", "Account", "Your name, password and account.",
                       name_form=name_form or NameForm(data={"name": current_user.name}),
                       pw_form=pw_form or ChangePasswordForm(), del_form=del_form or DeleteAccountForm(),
                       email=current_user.email, role=current_user.role,
                       created=current_user.created_at, password_rule=PASSWORD_RULE), status


@bp.get("/account")
@login_required
def account():
    return _account()


@bp.post("/account/name")
@login_required
def account_name():
    form = NameForm()
    if not form.validate_on_submit():
        return _account(name_form=form, status=400)
    current_user.name = form.name.data
    db.session.commit()
    flash("Name updated.", "ok")
    return redirect(url_for("auth.account"))


@bp.post("/account/password")
@login_required
@limiter.limit("10 per minute")
def account_password():
    form = ChangePasswordForm()
    if not form.validate_on_submit():
        return _account(pw_form=form, status=400)
    if not check_password_hash(current_user.password_hash, form.current.data):
        form.current.errors.append("Current password is incorrect.")
        return _account(pw_form=form, status=400)
    user = current_user._get_current_object()
    user.password_hash = hash_password(form.password.data)
    user.must_change_password = False
    _rotate_and_relogin(user)
    audit("password_change", f"user:{user.id}")
    flash("Password changed. Other devices have been signed out.", "ok")
    return redirect(url_for("auth.account"))


@bp.post("/account/delete")
@login_required
@limiter.limit("10 per minute")
def account_delete():
    form = DeleteAccountForm()
    if not form.validate_on_submit():
        return _account(del_form=form, status=400)
    user = current_user._get_current_object()
    if not check_password_hash(user.password_hash, form.password.data):
        form.password.errors.append("Password is incorrect.")
        return _account(del_form=form, status=400)
    if is_last_active_admin(user):
        form.password.errors.append("You are the only active admin. Promote another admin first.")
        return _account(del_form=form, status=400)
    uid = user.id
    audit("account_delete", f"user:{uid}")
    logout_user()
    session.clear()
    db.session.delete(user)                # profile and settings rows go with it
    db.session.commit()
    flash("Your account and all its data were deleted.", "ok")
    return redirect(url_for("auth.login"))


@bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password_forced():
    """Shown after an admin set a temporary password."""
    if not current_user.must_change_password:
        return redirect(url_for("auth.account"))
    form = ForcedPasswordForm()
    if form.validate_on_submit():
        user = current_user._get_current_object()
        if check_password_hash(user.password_hash, form.password.data):
            form.password.errors.append("Choose a new password, not the temporary one.")
        else:
            user.password_hash = hash_password(form.password.data)
            user.must_change_password = False
            _rotate_and_relogin(user)
            audit("password_change", f"user:{user.id}")
            return redirect("/")
    return render_template("auth/change_password.html", form=form, user=None,
                           settings={"theme": "light"}, password_rule=PASSWORD_RULE)
