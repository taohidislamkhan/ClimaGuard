"""Flask-WTF forms (each carries a CSRF token)."""

from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import BooleanField, EmailField, HiddenField, PasswordField, StringField
from wtforms.validators import DataRequired, EqualTo, Length, Regexp, ValidationError

from .security import password_problem

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def strong_password(_form, field):
    msg = password_problem(field.data or "")
    if msg:
        raise ValidationError(msg)


def _lower(v):
    return (v or "").strip().lower()


def _strip(v):
    return (v or "").strip()


class LoginForm(FlaskForm):
    email = EmailField("Email", filters=[_lower], validators=[DataRequired(), Length(max=254)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=200)])
    remember = BooleanField("Remember me")
    next = HiddenField()


class SignupForm(FlaskForm):
    name = StringField("Name", filters=[_strip], validators=[DataRequired(), Length(max=60)])
    email = EmailField("Email", filters=[_lower], validators=[
        DataRequired(), Length(max=254), Regexp(EMAIL_RE, message="Enter a valid email address.")])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=200), strong_password])
    confirm = PasswordField("Confirm password", validators=[
        DataRequired(), EqualTo("password", message="Passwords do not match.")])


class NameForm(FlaskForm):
    name = StringField("Name", filters=[_strip], validators=[DataRequired(), Length(max=60)])


class ChangePasswordForm(FlaskForm):
    current = PasswordField("Current password", validators=[DataRequired(), Length(max=200)])
    password = PasswordField("New password", validators=[DataRequired(), Length(max=200), strong_password])
    confirm = PasswordField("Confirm new password", validators=[
        DataRequired(), EqualTo("password", message="Passwords do not match.")])


class ForcedPasswordForm(FlaskForm):
    password = PasswordField("New password", validators=[DataRequired(), Length(max=200), strong_password])
    confirm = PasswordField("Confirm new password", validators=[
        DataRequired(), EqualTo("password", message="Passwords do not match.")])


class DeleteAccountForm(FlaskForm):
    password = PasswordField("Your password", validators=[DataRequired(), Length(max=200)])


class LogoutForm(FlaskForm):
    pass
