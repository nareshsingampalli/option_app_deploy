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
NSE_MARKET_START = "09:15"
NSE_MARKET_END   = "15:15"
MCX_MARKET_START = "09:00"
MCX_MARKET_END   = "23:30"

# ── Data Fetching Configuration ──────────────────────────────────────────────
CANDLE_INTERVAL_MINUTES = 3    # Single source of truth for fetchers, scheduler, and chart labels

# ── Background scheduler hours ───────────────────────────────────────────────
SCHEDULER_HOURS: dict[str, dict] = {
    "NSE": {"start": "09:15:20", "end": "15:40:00", "prefix": "option"},
    "BSE": {"start": "09:15:20", "end": "15:40:00", "prefix": "option"},
    "MCX": {"start": "09:15:20", "end": "23:59:00", "prefix": "mcx"},
}

# ── Strike selection ─────────────────────────────────────────────────────────
DEFAULT_NUM_STRIKES = 3      # strikes on each side of ATM
MCX_STRIKE_STEP     = 50     # ATM rounding step for MCX

# ── API configuration ────────────────────────────────────────────────────────
# Single source of truth for the Upstox API credentials.
# The dashboard uses /api/refresh-token to reload these from the .env file at runtime.
UPSTOX_API_URL      = os.getenv("UPSTOX_API_URL", "https://api.upstox.com")
UPSTOX_ACCESS_TOKEN =  "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI4Q0FRNzUiLCJqdGkiOiI2OWIyM2I4MWE5YzAwZDAwNWIzMWJkMmQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzczMjg4MzIxLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzMzNTI4MDB9.CoCojyphbBuGxh6NGo_DLnRdSxTRziyVU6Fjgh7iriY"
# Location of the .env file (used by refresh-token endpoint)
ENV_FILE = os.getenv("ENV_FILE") or "/home/ubuntu/refactor_app/.env"
if not os.path.exists(ENV_FILE):
    _local = os.path.join(os.getcwd(), ".env")
    if os.path.exists(_local):
        ENV_FILE = _local

def reload_access_token():
    """Reloads the token from the environment into the live config variable."""
    global UPSTOX_ACCESS_TOKEN
    new_token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if new_token:
        UPSTOX_ACCESS_TOKEN = new_token
        return True
    return False

# ── API rate limits ──────────────────────────────────────────────────────────
UPSTOX_RATE_LIMIT_CALLS  = 15
UPSTOX_RATE_LIMIT_PERIOD = 1.0   # seconds
