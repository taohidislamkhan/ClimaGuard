"""Flask extension instances, bound to the app in ``create_app``."""

from __future__ import annotations

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()
# In-memory counters: fine for one process (see README, known limitations).
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://", default_limits=[])
