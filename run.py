"""
run.py — single entry point for the Option Chain Dashboard.

Usage:
    python run.py
"""

# Import app first, then register routes + scheduler (avoids circular imports)
from dashboard import app, socketio
import dashboard.routes     # noqa: F401
import dashboard.scheduler  # noqa: F401
from dashboard.scheduler import start_schedulers

import threading
_scheduler_started = False
_scheduler_lock = threading.Lock()

def _ensure_schedulers():
    global _scheduler_started
    with _scheduler_lock:
        if not _scheduler_started:
            print("[System] Initializing production-ready master schedulers...")
            start_schedulers()
            _scheduler_started = True

# Start schedulers immediately on import (required for WSGI servers like Gunicorn)
_ensure_schedulers()

if __name__ == "__main__":
    print("Starting Option Chain Dashboard (Development Mode) on port 8010...")
    socketio.run(app, host="0.0.0.0", port=8010, debug=False, allow_unsafe_werkzeug=True)
