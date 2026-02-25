# Option App — Low-Level Design (LLD)

> **Last Updated:** 2026-02-25  
> **Author:** Auto-generated from source analysis  
> **Stack:** Python 3, Flask, Flask-SocketIO (eventlet), Upstox SDK, Pandas, SciPy

---

## 1. System Overview

The **Option App** is a real-time and historical **NSE / MCX option chain analysis dashboard**.  
It is a Flask-based web application deployed on an Ubuntu VM that:

- Fetches **live and historical option chain data** from the **Upstox API**
- Calculates **Implied Volatility (IV)**, **Rate-of-Change (ROC)**, and other derivatives
- Serves data via **REST API** and pushes real-time updates via **WebSocket (Socket.IO)**
- Renders an interactive web dashboard (Highcharts-based) for visual analysis

---

## 2. Architecture Overview

### 2.1 Component Map

| Component | File | Role |
|---|---|---|
| Flask Dashboard | `option_dashboard.py` | HTTP server, WebSocket hub, background scheduler |
| NSE Option Chain | `option_chain.py` | Full data pipeline for Nifty 50 (NSE) options |
| MCX Option Chain | `option_chain_mcx.py` | Full data pipeline for MCX commodity options |
| Candle Fetchers | `candle_fetchers.py` | Upstox API wrappers (intraday, historical, expired) |
| Frontend Template | `templates/option_comparison.html` | Highcharts-based interactive UI |

### 2.2 End-to-End Data Flow

```
┌────────────────────────────────────────────┐
│           option_dashboard.py              │
│                                            │
│  ┌──────────────────────┐                  │
│  │  Background Scheduler│  (daemon thread) │
│  │  NSE: min%5==1        │  09:15–15:40    │
│  │  MCX: min%5==1        │  09:15–23:59    │
│  └──────────┬───────────┘                  │
│             │ subprocess.run(--live)        │
│             ▼                              │
│  option_chain.py / option_chain_mcx.py     │
│             │                              │
│             ▼   (Upstox API calls)         │
│  ┌──────────────────────┐                  │
│  │  candle_fetchers.py  │                  │
│  └──────────────────────┘                  │
│             │                              │
│             ▼  writes                      │
│  ┌──────────────────────────────────┐      │
│  │  option_data_tabular_YYYY-MM-DD  │.csv  │
│  │  option_meta_YYYY-MM-DD          │.json │
│  └──────────────────────────────────┘      │
│             │                              │
│             ▼  reads                       │
│  GET /api/option-data                      │
│             │                              │
│             ▼  Socket.IO emit              │
│  Frontend (Highcharts) ◄── data_updated    │
└────────────────────────────────────────────┘
```

---

## 3. Module-Level Design

---

### 3.1 `option_dashboard.py` — Flask Application

#### Responsibilities
- Serve the frontend HTML template at `GET /`
- Expose REST API: `GET /api/option-data`
- Expose token refresh: `POST /api/refresh-token`
- Manage per-symbol background scheduler daemon threads
- Broadcast WebSocket events when fresh data is written

#### Configuration

```python
MARKET_HOURS = {
    'NSE': ('09:15:20', '15:40:00', 'option_chain.py',     'option'),
    'MCX': ('09:15:20', '23:59:00', 'option_chain_mcx.py', 'mcx'),
}
```

#### Key Functions

---

**`_symbol_scheduler(symbol, start_s, end_s, script)`**  
- Runs as an independent daemon thread per symbol (NSE and MCX never block each other)
- Pre-market: sleeps until `start_s` (market open)
- In-market: checks `now.minute % 5 == 1` every 15 seconds; triggers fetch once per qualifying minute
- Post-market: sleeps until next day's `start_s`
- Uses `last_fetch_min` guard to prevent double-fetches within the same minute

```
State machine:
  cur_secs < start_secs → sleep(wait)          [pre-market]
  cur_secs > end_secs   → sleep(tomorrow)      [post-market]
  minute % 5 == 1       → _run_fetch()         [in-market]
  else                  → sleep(15s)           [in-market, waiting]
```

---

**`_run_fetch(symbol, script)`**  
- Protected by `_fetch_locks[symbol]` (`threading.Lock`)
- Uses `lock.acquire(blocking=False)` → skips if already running
- Executes: `subprocess.run([python, script, today, "--live"], timeout=280)`
- On success: emits `data_updated` Socket.IO event with `{ prefix, timestamp }`

---

**`get_option_data()` — `GET /api/option-data`**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `date` | str | `2026-02-20` | Target date `YYYY-MM-DD` |
| `time` | str | `""` | Optional time `HH:MM` for snapshot |
| `live` | bool | `false` | If true, forces today + ignores time slider |
| `symbol` | str | `NSE` | `NSE` or `MCX` |

**Logic flow:**
1. Resolve filename: `{prefix}_data_tabular_{date}[_{time}].csv`
2. If CSV missing AND not today's live data → set `needs_fetch = True`
3. Check meta JSON for existing auth errors → short-circuit if `Invalid token`
4. If `needs_fetch`: acquire lock → subprocess fetch → release lock
5. Post-fetch: if CSV still missing, return error from meta
6. Read CSV → read meta → fallback spot price from CSV if meta lacks it
7. Return `{ data: [...records], meta: {...} }`

**Output file naming:**
```
option_data_tabular_2026-02-25.csv        ← live/latest
option_data_tabular_2026-02-25_1030.csv   ← time-specific snapshot
option_meta_2026-02-25.json
option_meta_2026-02-25_1030.json
mcx_data_tabular_2026-02-25.csv
mcx_meta_2026-02-25.json
```

---

**`refresh_token()` — `POST /api/refresh-token`**
- Calls `load_dotenv("/home/ubuntu/refactor_app/.env", override=True)`
- If meta has an `Invalid token` error, clears it so the next scheduler cycle can retry

---

### 3.2 `option_chain.py` — NSE Option Chain Pipeline

#### Design Pattern: Strategy + Context

```
MarketDataStrategy (ABC)
│   get_spot_price(date_str, time_str) → float
│   get_instruments(spot_price, date_str) → (instruments, expiry, is_expired)
│   get_iv_spot_data(date_str) → DataFrame
│   get_candle_data(instrument, from_date, to_date) → DataFrame
│
├── LiveStrategy
│   └── Uses: IntradayCandleFetcher
│   └── Spot: 5-min intraday → fallback 1-min (pre 09:20)
│   └── Instruments: get_option_chain_instruments()
│
├── HistoricalStrategy
│   └── Uses: HistoricalCandleFetcher
│   └── Spot: 5-min for target date (if time_str) → fallback daily
│   └── Instruments: get_option_chain_instruments()
│
└── ExpiredStrategy
    └── Uses: ExpiredCandleFetcher + HistoricalCandleFetcher (for index spot)
    └── Spot: hist_fetcher for index (not in Expired API), 5-min → fallback daily
    └── Instruments: get_expired_option_chain_instruments()
    └── Candles: formats key as "key|DD-MM-YYYY" before fetch

OptionChainProcessor (Context)
    └── strategy: MarketDataStrategy
    └── run(date_str, time_str) → orchestrates all steps
    └── process_data(df, spot_map, instr) → derived indicators
    └── save_results(…) → writes CSV
    └── save_meta(…) → writes JSON
```

#### Strategy Selection (in `main()`)

```
--live flag OR (today's date AND no time_str)
    → LiveStrategy

else:
    fetch last_expiry from ExpiredCandleFetcher.fetch_expiries()
    target_date <= last_expiry → ExpiredStrategy
    else                       → HistoricalStrategy
```

---

#### `get_option_chain_instruments(spot_price, num_strikes=3, reference_date)` — Active Instruments

1. Fetches `https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz`
2. Filters: `instrument_type == 'OPTIDX'` AND `name == 'NIFTY'`
3. Renames: `tradingsymbol → trading_symbol`, `strike → strike_price`
4. Finds nearest expiry `>= reference_date`
5. Selects `num_strikes` strikes on each side of ATM
6. Returns list of `{ key, symbol, strike, type (CE/PE), expiry }`

Returns `([], None, True)` if all expiries are in the past (expired).

---

#### `get_expired_option_chain_instruments(spot_price, num_strikes=3, reference_date)` — Expired Instruments

1. Uses `ExpiredCandleFetcher.fetch_expiries(NSE_INDEX|Nifty 50)`
2. Finds first expiry `>= target_date`
3. Fetches contracts: `ExpiredCandleFetcher.fetch_contracts(underlying, expiry_str)`
4. Builds DataFrame, filters CE/PE by nearest strikes
5. Returns list with same shape as active instruments

---

#### `OptionChainProcessor.process_data(df, spot_map, instr)`

| Step | Operation |
|---|---|
| 1 | Rename: `close → ltp`, `open_interest → oi` |
| 2 | `change_in_oi = oi.diff()` |
| 3 | IV via Black-Scholes Newton-Raphson per candle row |
| 4 | `change_in_ltp = ltp.diff()` |
| 5 | `roc_oi = oi.pct_change() * 100` |
| 6 | `roc_volume = volume.pct_change() * 100` |
| 7 | `roc_iv = iv.pct_change() * 100` |
| 8 | `coi_vol_ratio = change_in_oi / volume` |
| 9 | Map `spot_price` from `spot_map[timestamp]` |
| 10 | Filter time: 09:15–15:15 |
| 11 | Return as `list[dict]` |

---

#### Black-Scholes / IV Functions

```python
black_scholes_call(S, K, T, r, sigma)  # standard BS call formula
black_scholes_put(S, K, T, r, sigma)   # standard BS put formula

implied_volatility(price, S, K, T, r, option_type)
  # Newton-Raphson: 100 iterations
  # T in years (seconds / 365*24*3600)
  # sigma clipped to [0.001, ∞)
  # risk-free rate r = 0.10 (10% annualised)
  # expiry time assumed 15:30 IST on expiry date
```

---

### 3.3 `option_chain_mcx.py` — MCX Option Chain Pipeline

Mirrors the NSE pipeline but with MCX-specific differences:

#### Key Differences vs NSE

| Aspect | NSE (`option_chain.py`) | MCX (`option_chain_mcx.py`) |
|---|---|---|
| Spot key | `NSE_INDEX\|Nifty 50` | `MCX_FO\|472789` (CrudeOil Mar 26 Future) |
| Instruments source | Upstox NSE CSV (`.gz`) | Upstox MCX JSON (`.gz`) |
| Strike rounding | Nearest from exchange CSV | `ATM = round(spot/50)*50`, fixed 50-pt step |
| Symbol filter | `OPTIDX` + `name==NIFTY` | 2-step: broad `startswith` + precise regex |
| Expiry rule | Nearest `>= reference_date` | **D-1 rule**: cutoff = `reference_date + 1 day` |
| Market hours filter | 09:15–15:15 | 09:00–23:30 |
| Multi-commodity | ❌ | ✅ CRUDEOIL, SILVER, NATURALGAS, GOLD |
| Expiry time for IV | 15:30 | 23:30 |
| IV iterations | 100 | 20 |

---

#### `get_option_chain_instruments(spot_price, num_strikes=5, reference_date, symbol='CRUDEOIL')`

**Step 1 — Broad filter (to determine expiry month)**
```python
broad = df[
    df['instrument_type'].isin(['CE', 'PE']) &
    df['trading_symbol'].str.startswith(symbol + ' ')
]
effective_cutoff = ref_dt + 1 day  # D-1 rule
target_expiry = first expiry > effective_cutoff
```

**Step 2 — Precise regex filter**
```python
precise_pattern = rf"^{commodity} (FUT \d{{1,2}}|\d+ (CE|PE) \d{{1,2}}) {MONTH} {YEAR}$"
# e.g. "^CRUDEOIL (\d+ (CE|PE) \d{1,2}) MAR 26$"
# matches: "CRUDEOIL 6600 CE 17 MAR 26"
```

Builds `expiry_map`: `{ 'CRUDEOIL': '17 MAR 26' }` — parsed from trading symbol suffix after `CE`/`PE`.

Returns list with extra field `expiry_str` (human-readable, for expired key formatting).

---

#### `ExpiredStrategy._format_expired_key(key, expiry_dt)`
```python
# Appends expiry date for Expired API compatibility
# Input:  "MCX_FO|12345"
# Output: "MCX_FO|12345|17-03-2026"
```

---

### 3.4 `candle_fetchers.py` — Upstox SDK Wrappers

#### Class Hierarchy

```
BaseCandleFetcher
│   __init__(access_token)  ← reads UPSTOX_ACCESS_TOKEN from env
│   _process_response(response) → DataFrame
│       columns: [timestamp, open, high, low, close, volume, open_interest]
│       index: datetime (sorted ascending)
│
├── IntradayCandleFetcher
│   └── fetch(instrument_key, timeframe, interval_num)
│       └── history_api.get_intra_day_candle_data(key, unit, interval)
│
├── HistoricalCandleFetcher
│   ├── _fetch_single(key, unit, interval_num, to_date, from_date)
│   │   └── history_api.get_historical_candle_data1(...)
│   │   └── SIGALRM 90s timeout (Linux only)
│   │
│   ├── fetch(key, timeframe, interval_num, lookback_days=90)
│   │   └── For minutes/hours + >30 days → _fetch_chunked()
│   │   └── For days/weeks/months → _fetch_single()
│   │
│   ├── _fetch_chunked(key, unit, interval_num, lookback_days, chunk_size=30)
│   │   └── Iterates backward in 30-day windows
│   │   └── Concatenates all chunks, deduplicates, sorts
│   │   └── Sleeps 500ms between chunks (rate limit safety)
│   │
│   └── fetch_3months(key, timeframe, interval_num)
│       └── Convenience: lookback_days=90
│
└── ExpiredCandleFetcher
    ├── __init__  ← also initializes ExpiredInstrumentApi
    ├── fetch_expiries(underlying_key) → list[date]
    │   └── expired_api.get_expiries(instrument_key)
    ├── fetch_contracts(underlying_key, expiry_date) → list[contract]
    │   └── expired_api.get_expired_option_contracts(instrument_key, expiry_date)
    └── fetch_candle_data(instrument_key, interval_str, to_date, from_date) → DataFrame
        └── expired_api.get_expired_historical_candle_data(...)
        └── interval_str: '1minute', '5minute', '30minute', 'day'
```

---

## 4. Data Schemas

### 4.1 CSV Output (`option_data_tabular_YYYY-MM-DD.csv`)

| Column | Type | Description |
|---|---|---|
| `date` | str (datetime) | 5-minute candle timestamp (IST) |
| `symbol` | str | Trading symbol e.g. `NIFTY24JAN23000CE` |
| `ltp` | float | Last traded price (candle close) |
| `change_in_ltp` | float | Delta from previous candle's LTP |
| `roc_oi` | float | OI rate-of-change `%` |
| `roc_volume` | float | Volume rate-of-change `%` |
| `roc_iv` | float | IV rate-of-change `%` |
| `coi_vol_ratio` | float | Change-in-OI / Volume |
| `spot_price` | float | Underlying spot price at this candle |

### 4.2 Metadata JSON (`option_meta_YYYY-MM-DD.json`)

```json
{
  "spot_price": 22500.0,
  "target_date": "2026-02-25",
  "target_time": null,
  "expiry_date": "2026-02-27",
  "fetched_at": "2026-02-25T10:01:00.123456",
  "has_data": true,
  "expired_contracts": false,
  "error": null
}
```

| Field | Description |
|---|---|
| `spot_price` | Underlying index/future price used for instrument selection |
| `target_date` | The date for which data was fetched |
| `target_time` | Optional HH:MM (null for daily snapshot) |
| `expiry_date` | The option expiry date used |
| `fetched_at` | ISO timestamp of when the fetch ran |
| `has_data` | Whether a CSV with rows was written |
| `expired_contracts` | True if the data is sourced from the Expired API |
| `error` | Error string if fetch failed, null otherwise |

---

## 5. Key Design Decisions

### 5.1 Strategy Pattern for Data Sources
The app must handle 3 distinct data regimes — **Live**, **Historical**, and **Expired** — each backed by different Upstox endpoints. The Strategy pattern isolates this complexity, keeping `OptionChainProcessor.run()` agnostic of regime.

### 5.2 Independent Scheduler Threads per Symbol
NSE closes at 15:40 while MCX runs until 23:59. Using independent daemon threads ensures one market's sleep/wait cycle never blocks the other.

### 5.3 File-Based Caching (CSV + JSON)
Preprocessed data is persisted to disk rather than held in memory. Benefits:
- Dashboard restarts without re-fetching all data
- Historical time-specific snapshots served instantly on repeat requests
- Crash recovery — data already on disk is immediately available

### 5.4 Per-Symbol Mutex for Concurrent Fetch Prevention
`threading.Lock` per symbol prevents the scheduler thread and a concurrent API request from both triggering a subprocess fetch simultaneously, avoiding duplicate Upstox API calls and file corruption.

### 5.5 D-1 Expiry Rule (MCX)
MCX options in the last day before expiry are highly illiquid. The app proactively switches to next month's contracts one day early to ensure the displayed data is meaningful.

### 5.6 Dual-Resolution Spot Fetch (5-min → 1-min fallback)
Between 09:15–09:20, 5-minute candle data may not yet exist. A 1-minute fallback ensures the very first fetch of the day still succeeds.

### 5.7 SIGALRM Timeout in HistoricalCandleFetcher
Upstox API calls can occasionally hang. A 90-second `SIGALRM` alarm is set before each `_fetch_single` call (on Linux/VM) to prevent the scheduler thread from hanging indefinitely.

### 5.8 Auth Error Short-Circuit
If an `Invalid token` error is recorded in the metadata JSON, subsequent API requests skip the fetch attempt entirely and return the error immediately, preventing repeated failed subprocess launches until the token is refreshed via `POST /api/refresh-token`.

---

## 6. API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Render `option_comparison.html` dashboard |
| `GET` | `/api/option-data` | Fetch option chain data (see params below) |
| `POST` | `/api/refresh-token` | Reload `.env` and clear auth error state |

### `GET /api/option-data` Parameters

| Param | Values | Description |
|---|---|---|
| `symbol` | `NSE` / `MCX` | Which market to fetch |
| `date` | `YYYY-MM-DD` | Target date |
| `time` | `HH:MM` | Optional intraday snapshot time |
| `live` | `true` / `false` | Force live-mode (today, no time filtering) |

### WebSocket Events

| Event | Direction | Payload |
|---|---|---|
| `connect` | Client → Server | (implicit, logs `request.sid`) |
| `disconnect` | Client → Server | (implicit, logs `request.sid`) |
| `data_updated` | Server → Client | `{ prefix: 'NSE'/'MCX', timestamp: ISO }` |

---

## 7. Deployment

| Item | Value |
|---|---|
| OS | Ubuntu VM (20.04+) |
| Port | `8010` |
| WSGI | **eventlet** (via `flask-socketio` async_mode) |
| Run command | `python3 option_dashboard.py` |
| Env file | `/home/ubuntu/refactor_app/.env` |
| Token var | `UPSTOX_ACCESS_TOKEN` |
| Scheduler | Embedded daemon threads (no cron / Celery) |
| Logs | stdout — tagged with `[Scheduler-NSE]`, `[API]`, `[WebSocket]` prefixes |

---

## 8. File Tree

```
option_app/
├── option_dashboard.py          # Flask app + scheduler + API
├── option_chain.py              # NSE pipeline (Strategy + Processor)
├── option_chain_mcx.py          # MCX pipeline (Strategy + Processor)
├── candle_fetchers.py           # Upstox SDK wrappers
├── requirements.txt             # Python dependencies
├── .env                         # UPSTOX_ACCESS_TOKEN (gitignored)
├── templates/
│   └── option_comparison.html   # Highcharts frontend
├── option_data_tabular_*.csv    # Runtime output (NSE)
├── option_meta_*.json           # Runtime metadata (NSE)
├── mcx_data_tabular_*.csv       # Runtime output (MCX)
└── mcx_meta_*.json              # Runtime metadata (MCX)
```

---

## 9. Dependencies (`requirements.txt`)

```
flask
flask-socketio
eventlet
pandas
numpy
scipy
requests
upstox-python-sdk
python-dotenv
```
