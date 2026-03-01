"""
run.py — single entry point for the Option Chain Dashboard.

Usage:
    python run.py
"""

from dotenv import load_dotenv
load_dotenv()

# Import app first, then register routes + scheduler (avoids circular imports)
from dashboard import app, socketio
import dashboard.routes     # noqa: F401 — registers @app.route decorators
import dashboard.scheduler  # noqa: F401 — exposes _fetch_locks
from dashboard.scheduler import start_schedulers

if __name__ == "__main__":
    print("Starting Option Chain Dashboard on port 8010...")
    print("Starting schedulers (minute % 5 == 1 during market hours):")
    start_schedulers()
    socketio.run(app, host="0.0.0.0", port=8010, debug=False, allow_unsafe_werkzeug=True)
