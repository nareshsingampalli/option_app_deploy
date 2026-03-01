"""Central configuration — all magic constants live here."""

# ── Black-Scholes ────────────────────────────────────────────────────────────
RISK_FREE_RATE: float = 0.1          # Annual risk-free rate

# ── NSE index instrument keys (static — DO NOT change) ───────────────────────
NSE_INDEX_KEYS: dict[str, str] = {
    "NIFTY":     "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "FINNIFTY":  "NSE_INDEX|Nifty Fin Service",
}

# ── MCX commodity underlying keys ───────────────────────────────────────────
MCX_FUT_KEYS: dict[str, str] = {
    "CRUDEOIL":   "MCX_COM|294",
    "NATURALGAS": "MCX_COM|401",
    "SILVER":     "MCX_COM|115",
    "GOLD":       "MCX_COM|114",
}

# ── Instrument list download URLs ────────────────────────────────────────────
NSE_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
MCX_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz"
BSE_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"

CACHE_DIR = "cache"
# ── Trading time windows (IST, used for data filtering) ─────────────────────
NSE_MARKET_START = "09:15"
NSE_MARKET_END   = "15:15"
MCX_MARKET_START = "09:00"
MCX_MARKET_END   = "23:30"

# ── Background scheduler hours ───────────────────────────────────────────────
SCHEDULER_HOURS: dict[str, dict] = {
    "NSE": {"start": "09:15:20", "end": "15:40:00", "prefix": "option"},
    "MCX": {"start": "09:15:20", "end": "23:59:00", "prefix": "mcx"},
}

# ── Strike selection ─────────────────────────────────────────────────────────
DEFAULT_NUM_STRIKES = 3      # strikes on each side of ATM
MCX_STRIKE_STEP     = 50     # ATM rounding step for MCX

# ── API rate limits ──────────────────────────────────────────────────────────
UPSTOX_RATE_LIMIT_CALLS  = 15
UPSTOX_RATE_LIMIT_PERIOD = 1.0   # seconds
