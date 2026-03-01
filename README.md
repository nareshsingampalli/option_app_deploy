# Option App

Option App provides live and historical option-chain data processing and a small Flask
dashboard to visualize comparisons. It supports both NSE (Nifty) and MCX commodity
option chains and can fetch data via the Upstox APIs (live, historical, and expired).

## Quick Start
1. Install dependencies:

	 ```bash
	 pip install -r requirements.txt
	 ```

2. Provide environment variables (example `.env`):

	 - `UPSTOX_ACCESS_TOKEN` — Upstox API access token
	 - `SENTRY_DSN` (optional) — Sentry DSN for error reporting

3. Run the dashboard:

	 ```bash
	 python option_dashboard.py
	 # open http://localhost:8010
	 ```

4. Fetch data scripts (examples):

	 ```bash
	 # Historical Nifty data for 2026-02-24
	 python option_chain.py 2026-02-24

	 # Live MCX data (today)
	 python option_chain_mcx.py --live
	 ```

## Files & Responsibilities

- `option_dashboard.py`: Flask + SocketIO dashboard and background scheduler. Schedules
	periodic fetches for NSE and MCX and serves the UI at port 8010.
- `option_chain.py`: NIFTY option-chain processor. Supports three zones: Live, Historical,
	and Expired (uses `ExpiredCandleFetcher`). Produces CSV and meta JSON files named
	`option_data_tabular_<YYYY-MM-DD>[ _HHMM].csv` and `option_meta_<...>.json`.
- `option_chain_mcx.py`: MCX commodity option-chain processor (CRUDEOIL, NATURALGAS).
	Same output patterns as NSE but with `mcx_` prefixes.
- `candle_fetchers.py`: Upstox API wrapper providing `IntradayCandleFetcher`,
	`HistoricalCandleFetcher`, and `ExpiredCandleFetcher`. Converts API responses to
	pandas DataFrames and handles chunking / rate-limit concerns.
- `templates/option_comparison.html`: Frontend visualizer used by the dashboard.
- `deploy.bat` / `deploy.ps1`: (Optional) deployment helpers for Windows / PowerShell.

## Entrypoints

- Dashboard: `python option_dashboard.py` (serves UI and background schedulers)
- NSE processor: `python option_chain.py <YYYY-MM-DD> [HH:MM] [--live] [--symbol=NAME]`
- MCX processor: `python option_chain_mcx.py <YYYY-MM-DD> [HH:MM] [--live]`

## Dependencies
See `requirements.txt` (core: `flask`, `flask-socketio`, `eventlet`, `pandas`,
`requests`, `python-dotenv`, `upstox-python-sdk`, `scipy`, `numpy`, `pytz`, `sentry-sdk`).

## Notes
- Today's date is treated specially: the scripts require `--live` to fetch intraday
	live data; without `--live` requests for today are blocked by design.
- Output files are written to the working directory so run scripts from the desired
	output folder or adjust paths in code.

## Troubleshooting
- If you see `Invalid token` errors, use the dashboard endpoint `/api/refresh-token`
	to clear cached metadata before retrying.

---
Updated to reflect current code (Feb 26, 2026).
