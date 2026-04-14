
import os
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
# Initialize Sentry
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
    print("[Sentry] Dashboard monitoring enabled.")

from flask import Flask
from flask_socketio import SocketIO

# Resolve absolute template path so the app works from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES = os.path.join(_HERE, "..", "templates")

_STATIC = os.path.join(_HERE, "..", "static")

# Detect best async mode for SocketIO (prefers eventlet for production)
try:
    import eventlet
    _async_mode = "eventlet"
except ImportError:
    _async_mode = "threading"

app = Flask(__name__, template_folder=_TEMPLATES, static_folder=_STATIC, static_url_path="/static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_async_mode)

# NOTE: routes and scheduler are imported by run.py AFTER this module loads,
# which avoids the circular import that would occur if we imported them here.
