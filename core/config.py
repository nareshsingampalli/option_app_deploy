"""Central configuration — all magic constants live here."""

# ── Black-Scholes ────────────────────────────────────────────────────────────
RISK_FREE_RATE: float = 0.1          # Annual risk-free rate

# ── NSE index instrument keys (static — DO NOT change) ───────────────────────
NSE_INDEX_KEYS: dict[str, str] = {
    "NIFTY":     "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "FINNIFTY":  "NSE_INDEX|Nifty Fin Service",
}

# ── BSE index instrument keys (static) ───────────────────────────────────────
BSE_INDEX_KEYS: dict[str, str] = {
    "SENSEX":    "BSE_INDEX|SENSEX",
}

# ── MCX commodity underlying keys ───────────────────────────────────────────
MCX_FUT_KEYS: dict[str, str] = {
    "CRUDEOIL":   "MCX_COM|294",
    "NATURALGAS": "MCX_COM|401",
    "SILVER":     "MCX_COM|115",
    "GOLD":       "MCX_COM|114",
}

import os
import threading
import redis

# ── Instrument list download URLs ────────────────────────────────────────────
# NOTE: UPSTOX_INSTRUMENT_URL is only set in mock/test environments.
#       UPSTOX_API_URL is for the Upstox REST API (requires auth) and must
#       NOT be used for instrument CDN downloads (assets.upstox.com is public).
def _get_instrument_url(path: str) -> str:
    mock_url = os.getenv("UPSTOX_INSTRUMENT_URL")
    if mock_url:
        return f"{mock_url}/{path}"
    return f"https://assets.upstox.com/{path}"

NSE_INSTRUMENT_URL = _get_instrument_url("market-quote/instruments/exchange/NSE.csv.gz")
MCX_INSTRUMENT_URL = _get_instrument_url("market-quote/instruments/exchange/MCX.json.gz")
BSE_INSTRUMENT_URL = _get_instrument_url("market-quote/instruments/exchange/BSE.json.gz")

CACHE_DIR = "cache"
# ── Trading time windows (IST, used for data filtering) ─────────────────────
NSE_MARKET_START = "09:14"
NSE_MARKET_END   = "15:30"
MCX_MARKET_START = "09:00"
MCX_MARKET_END   = "23:50"

# ── Data Fetching Configuration ──────────────────────────────────────────────
CANDLE_INTERVAL_MINUTES = 15    # Single source of truth for fetchers, scheduler, and chart labels

# ── Background scheduler hours ───────────────────────────────────────────────
SCHEDULER_HOURS: dict[str, dict] = {
    "NSE": {"start": "09:15:20", "end": "15:40:00", "prefix": "option"},
    "BSE": {"start": "09:15:20", "end": "15:40:00", "prefix": "option"},
    "MCX": {"start": "09:00:20", "end": "23:59:00", "prefix": "mcx"},
}

# ── Strike selection ─────────────────────────────────────────────────────────
DEFAULT_NUM_STRIKES = 3      # strikes on each side of ATM
MCX_STRIKE_STEP     = 50     # ATM rounding step for MCX

# ── API configuration ────────────────────────────────────────────────────────
# Single source of truth for the Upstox API credentials.
# The dashboard uses /api/refresh-token to reload these from the .env file at runtime.
UPSTOX_API_URL      = os.getenv("UPSTOX_API_URL", "https://api.upstox.com")
UPSTOX_ACCESS_TOKEN =  "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI4Q0FRNzUiLCJqdGkiOiI2OWQ1YmMwMGVmMTNhOTdjYzIxZjRkMGIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzc1NjE0OTc2LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzU2ODU2MDB9.ZjEOYJSz6kHM6-08z11RjFRS6j6saPIvYWDBXoB1ORw"
# ── Token Refresh (Manual fallback) ──────────────────────────────────────────
def reload_access_token():
    """
    Manually re-sync the token from the environment as a fallback 
    if the Redis MQ listener is unreachable (common in VM setups).
    """
    from dotenv import load_dotenv
    load_dotenv(override=True)
    new_token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if new_token:
        global UPSTOX_ACCESS_TOKEN
        UPSTOX_ACCESS_TOKEN = new_token
        print(f"[API] Token manually reloaded from environment.")
        return True
    return False

# ── API rate limits ──────────────────────────────────────────────────────────
UPSTOX_RATE_LIMIT_CALLS  = 7
UPSTOX_RATE_LIMIT_PERIOD = 0.6   # seconds

# ── Redis Token Sync (MQ) ───────────────────────────────────────────────────
def _start_token_listener():
    """Starts a background thread to listen for token updates via Redis Pub/Sub."""
    def listen():
        try:
            # Connect to local Redis (assumes Redis is running on the same VM)
            password = "yourpassword123"
            r = redis.Redis(host='127.0.0.1', port=6379, db=0, password=password, decode_responses=True)
            p = r.pubsub()
            p.subscribe('upstox_token_updates')
            print("[MQ] Ready! Listening for token updates on 'upstox_token_updates' channel.")
            
            for message in p.listen():
                if message['type'] == 'message':
                    new_token = message['data']
                    if new_token:
                        global UPSTOX_ACCESS_TOKEN
                        UPSTOX_ACCESS_TOKEN = new_token
                        print(f"[MQ] Received new token. Live config updated via broadcast.")
        except Exception as e:
            print(f"[MQ-ERROR] Token listener failed to connect to Redis at 127.0.0.1:6379. "
                  f"Token sync will be disabled. Error: {e}")
            print(f"[MQ-INFO] Using fallback/hardcoded token. You can manually refresh via /api/refresh-token if you update your .env file.")

    # Ensure only one listener thread runs
    if not any(t.name == "UpstoxTokenListener" for t in threading.enumerate()):
        listener_thread = threading.Thread(target=listen, name="UpstoxTokenListener", daemon=True)
        listener_thread.start()

# Initialize the listener on module load
_start_token_listener()
