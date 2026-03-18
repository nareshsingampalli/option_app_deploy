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

# ── API rate limits ──────────────────────────────────────────────────────────
UPSTOX_RATE_LIMIT_CALLS  = 15
UPSTOX_RATE_LIMIT_PERIOD = 1.0   # seconds
