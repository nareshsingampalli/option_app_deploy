"""Dashboard routes — all Flask API endpoints."""

from __future__ import annotations

import json
import os
import sys
import subprocess
import time

import pandas as pd
from flask import jsonify, render_template, request
from datetime import datetime
from flask_socketio import join_room, leave_room, rooms

from dashboard import app, socketio
from core.utils import ist_now
from core.config import SCHEDULER_HOURS, NSE_INDEX_KEYS, MCX_FUT_KEYS


def _get_fetch_locks():
    """Lazily import _fetch_locks to avoid circular import at module load."""
    from dashboard.scheduler import _fetch_locks
    return _fetch_locks


@app.route("/")
def option_comparison():
    return render_template("option_comparison.html")


@app.route("/logs")
def view_logs():
    log_path = os.path.join(os.getcwd(), "app.log")
    if not os.path.exists(log_path):
        return "Log file not found.", 404
    try:
        with open(log_path) as f:
            return "<pre>" + "".join(f.readlines()[-50:]) + "</pre>"
    except Exception as e:
        return f"Error reading logs: {e}", 500


@app.route("/api/market-status")
def market_status():
    """
    Single source of truth for market hours.
    Returns whether the market is currently open for a given exchange,
    along with the configured trading window.
    Frontend should call this instead of duplicating timing logic in JS.
    """
    from core.utils import get_last_trading_day, TRADING_HOLIDAYS_2026

    exchange = request.args.get("exchange", "NSE").upper()

    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    start_str = cfg["start"][:5]   # "HH:MM"
    end_str   = cfg["end"][:5]     # "HH:MM"

    now      = ist_now()
    today_str = now.strftime("%Y-%m-%d")
    now_t    = now.time()

    start_t  = datetime.strptime(cfg["start"], "%H:%M:%S").time()
    end_t    = datetime.strptime(cfg["end"],   "%H:%M:%S").time()

    # Is today a valid trading day?
    trading_day = get_last_trading_day(now).strftime("%Y-%m-%d")
    is_trading_day = (trading_day == today_str)

    def secs(t):
        return t.hour * 3600 + t.minute * 60 + t.second

    is_open = is_trading_day and (secs(start_t) <= secs(now_t) <= secs(end_t))

    if not is_trading_day:
        reason = "weekend" if now.weekday() >= 5 else "holiday"
    elif secs(now_t) < secs(start_t):
        reason = "pre_market"
    elif secs(now_t) > secs(end_t):
        reason = "post_market"
    else:
        reason = "open"

    return jsonify({
        "exchange":   exchange,
        "is_open":    is_open,
        "reason":     reason,
        "start":      start_str,
        "end":        end_str,
        "now_ist":    now.strftime("%H:%M:%S"),
    })


@app.route("/api/refresh-token", methods=["POST"])
def refresh_token():
    try:
        import glob
        import core.config
        
        print(f"[API] Refreshing token state...")
        
        # Update the live config variable using the centralized helper
        # Since we are decoupled from .env, this just returns True now
        # as MQ handled the actual update.
        if core.config.reload_access_token():
            print(f"[API] Live config state verified.")
        else:
            print("[API] WARNING: Failed to verify live config state.")

        # Clear out any cached 'Invalid token' errors in all recent metadata files
        for dr in ["nse_data", "mcx_data"]:
            meta_files = glob.glob(os.path.join(os.getcwd(), dr, "*meta*.json"))
            for path in meta_files:
                try:
                    with open(path) as f:
                        meta = json.load(f)
                    if "error" in meta and ("token" in meta["error"].lower() or "auth" in meta["error"].lower() or "unauthorized" in meta["error"].lower()):
                        meta.pop("error", None)
                        with open(path, "w") as f:
                            json.dump(meta, f, indent=4)
                except Exception:
                    pass
                    
        return jsonify({"message": "Token refreshed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/option-data")
def get_option_data():
    from core.utils import ist_now, get_last_trading_day
    
    # If starting app on weekend, default to last working day
    default_date = get_last_trading_day().strftime("%Y-%m-%d")
    date_str    = request.args.get("date", default_date)

    # Always shift weekends to the preceding Friday
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    actual_dt = get_last_trading_day(target_dt)
    if actual_dt != target_dt:
        print(f"[API] Date {date_str} is weekend. Shifting to last working day: {actual_dt.strftime('%Y-%m-%d')}")
        date_str = actual_dt.strftime("%Y-%m-%d")

    time_str    = request.args.get("time", "")
    live_mode   = request.args.get("live", "false").lower() == "true"
    exchange    = request.args.get("exchange", "NSE").upper()
    symbol      = request.args.get("symbol", "NIFTY").upper()
    interval    = int(request.args.get("interval", "15"))

    print(f"[API] exch={exchange}, symbol={symbol}, date={date_str}, time={time_str}, live={live_mode}")

    is_today    = date_str == ist_now().strftime("%Y-%m-%d")
    prefix      = "mcx" if exchange == "MCX" else "option"

    # Auto-upgrade to live mode only when the market is OPEN
    if is_today and not live_mode:
        stat = market_status().get_json()
        if stat["is_open"]:
            print(f"[API] Date is today and market is OPEN, auto-upgrading to live mode.")
            live_mode = True
        else:
            print(f"[API] Date is today but market is CLOSED, proceeding with historical fetch.")

    # Store requested time for metadata extraction later
    requested_time = time_str if not live_mode else ""

    if live_mode or is_today:
        if live_mode:
            date_str = ist_now().strftime("%Y-%m-%d")
        # For today, we always use the main accumulating file. 
        # This prevents duplicate fetching and extra snapshot files.
        time_str = ""

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    suffix       = f"_{time_str.replace(':', '')}" if time_str else ""
    # Add interval to suffix to avoid overwriting files with different granularity
    suffix       += f"_int{interval}"
    sym          = symbol.lower()
    
    # Map prefix to directory name
    dir_name = "nse_data" if prefix == "option" else "mcx_data"
    
    csv_path     = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_tabular_{date_str}{suffix}.csv")
    meta_path    = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_meta_{date_str}{suffix}.json")

    file_exists  = os.path.exists(csv_path)
    file_age_s   = time.time() - os.path.getmtime(csv_path) if file_exists else 999999
    
    # ── Robust Needs-Fetch Logic ──────────────────────────────────────────────
    # We fetch if:
    # 1. File doesn't exist
    # 2. File exists but is empty (has_data: false) and not recently checked
    # 3. Market is open and data is > 5 min old
    # 4. Market is closed but we haven't done the "Final EOD Fetch" (after 15:40)
    
    needs_fetch = not file_exists
    cached_meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                cached_meta = json.load(f)
        except: pass

    has_valid_data = cached_meta.get("has_data", False)
    is_fallback    = cached_meta.get("is_fallback", False)
    file_mtime     = os.path.getmtime(csv_path) if file_exists else 0
    meta_mtime     = os.path.getmtime(meta_path) if os.path.exists(meta_path) else 0
    last_check_age = time.time() - max(file_mtime, meta_mtime)

    if file_exists:
        if live_mode or is_today:
            now_dt = ist_now()
            # Market Ends: 15:40 NSE/BSE, 23:40 MCX
            m_end_h, m_end_m = (15, 40) if exchange != "MCX" else (23, 40)
            m_end_dt = now_dt.replace(hour=m_end_h, minute=m_end_m, second=0, microsecond=0)
            is_closed = now_dt > m_end_dt
            
            if is_closed:
                # If market is closed, check if we have data from AFTER the close
                # If yes, it's final EOD data — servable forever.
                if max(file_mtime, meta_mtime) < m_end_dt.timestamp():
                    # We only have mid-day data. Need one final fetch.
                    needs_fetch = True
                else:
                    # We have EOD data.
                    needs_fetch = False
            else:
                # During market hours — 5 min TTL
                needs_fetch = last_check_age > 300
        else:
            # Historical (Past Date)
            if is_fallback:
                # If it's a fallback, retry once an hour to see if real data appeared
                needs_fetch = last_check_age > 3600
            elif not has_valid_data:
                # If it failed to get data (holiday/error), retry every 30 min
                needs_fetch = last_check_age > 1800
            else:
                # Correct historical data never changes
                needs_fetch = False

    # Skip fetch if auth error cached in meta
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                cached_meta = json.load(f)
            if "Invalid token" in cached_meta.get("error", ""):
                return jsonify({"error": cached_meta["error"], "meta": cached_meta}), 200
        except Exception:
            pass

    if needs_fetch:
        from dashboard.scheduler import get_lock
        lock = get_lock(symbol)
        
        if lock.acquire(blocking=False):
            print(f"[API] Data stale/missing for {symbol}. Starting background refresh...")
            # Proactively notify the frontend via WebSocket that we are starting a fetch
            socketio.emit("data_fetching", {"symbol": symbol, "message": f"Processing {date_str} data for {symbol}...\u2026"}, room=symbol)
            
            def bg_fetch_task():
                from datetime import datetime, timedelta
                curr_date = date_str
                retries = 0
                max_fallback = 5 
                
                while retries < max_fallback:
                    suffix = f"_{time_str.replace(':', '')}" if time_str else ""
                    suffix += f"_int{interval}"
                    c_path = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_tabular_{curr_date}{suffix}.csv")
                    m_path = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_meta_{curr_date}{suffix}.json")
                    
                    try:
                        # Only fetch if file is missing or old
                        if not os.path.exists(c_path) or (time.time() - os.path.getmtime(c_path) > 60):
                            print(f"[API-BG] Fetching {symbol} for {curr_date}...")
                            from storage.db_storage import build_storage_chain
                            from strategies.handlers import build_pipeline
                            
                            storage  = build_storage_chain()
                            pipeline = build_pipeline(exchange, curr_date, live_mode, symbol, storage, interval=interval)
                            
                            if pipeline:
                                pipeline.run(symbol, curr_date, time_str)
                                
                                # Check if it was a holiday (404) or completely empty payload
                                if getattr(pipeline.fetcher, "last_status", None) == 404 or not os.path.exists(c_path) or os.path.getsize(c_path) == 0:
                                    from core.utils import get_last_trading_day
                                    print(f"[API-BG] {curr_date} has no data (404/Empty). Immediately switching to previous trading day...")
                                    dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=1)
                                    dt = get_last_trading_day(dt)
                                    curr_date = dt.strftime("%Y-%m-%d")
                                    retries += 1
                                    continue
                                
                                # Success
                                socketio.emit("data_updated", 
                                              {"prefix": exchange, "symbol": symbol, "date": curr_date, "timestamp": datetime.now().isoformat()},
                                              room=symbol)
                                break
                            else:
                                print(f"[API-BG] Pipeline blocked for {curr_date}.")
                                break
                        else:
                            # File already exists and is fresh
                            socketio.emit("data_updated", 
                                          {"prefix": exchange, "symbol": symbol, "date": curr_date, "timestamp": datetime.now().isoformat()},
                                          room=symbol)
                            break
                    except Exception as e:
                        print(f"[API-BG] Error: {e}")
                        break
                if retries >= max_fallback:
                    socketio.emit("market_status", 
                                  {"symbol": symbol, "exchange": exchange, "status": "unavailable", "message": f"Historical data from Upstox is currently empty or unavailable. Please try again later."},
                                  room=symbol)

                try:
                    lock.release()
                except: pass
            
            import threading
            threading.Thread(target=bg_fetch_task, daemon=True).start()
        else:
            print(f"[API] Fetch already in progress for {symbol}. Serving existing data...")

        if not os.path.exists(csv_path):
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("error"):
                    return jsonify({"error": meta["error"], "meta": meta}), 200
                if meta.get("expired_contracts"):
                    return jsonify({"error": f"Contracts for {date_str} have expired.", "meta": meta}), 200
                return jsonify({"data": [], "meta": meta}), 200
            
            # Clear information instead of a generic error
            msg = f"Waiting for data from Upstox server for {symbol}. Fetch is in progress..."
            return jsonify({"error": msg}), 200

    try:
        df   = pd.read_csv(csv_path)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        # For "Historical view of today", filter the master dataframe to only show candles up to the requested time.
        # This allows the slider to work without creating extra snapshot files.
        if requested_time and not df.empty:
            try:
                if "date_dt" not in df.columns:
                    df["date_dt"] = pd.to_datetime(df["date"])
                
                # Make target aware or naive to match dataframe
                tgt = pd.to_datetime(f"{date_str} {requested_time}")
                
                # Comparison safety: ensure both are naive for comparison
                # We use .dt.tz_localize(None) because Upstox dates are often +05:30 aware
                df_dt = pd.to_datetime(df["date"], errors='coerce')
                df_naive = df_dt.dt.tz_localize(None)
                tgt_naive = tgt.tz_localize(None) if tgt.tzinfo else tgt
                
                # Filter df to only include data up to the requested slider time
                # Drop rows where date parsing failed
                df = df[df_naive.notnull() & (df_naive <= tgt_naive)].copy()
            except Exception as e:
                print(f"[API] Time filter failed: {e}")

        # Recover spot price from CSV if missing in meta OR if we are looking at a specific time
        if not df.empty and (not meta.get("spot_price") or requested_time):
            try:
                # In our filtered (or original) DF, the last record is the closest to the target time
                last_spot = df["spot_price"].iloc[-1]
                if pd.notnull(last_spot):
                    meta["spot_price"] = float(last_spot)
            except Exception as e:
                print(f"[API] spot recovery failed: {e}")

        if meta.get("expired_contracts") and not meta.get("has_data"):
            return jsonify({"error": f"Archived contracts for {date_str} are not available.", "meta": meta})

        # Replace NaNs and Infinity with concrete values to avoid invalid JSON (browser JSON.parse rejects them)
        import numpy as np
        # Converting all NaNs to "" is the safest for the browser to avoid crashes
        df_clean = df.fillna("")
        records = df_clean.to_dict(orient="records")
        
        # Also clean up meta
        def clean_meta(d):
            if isinstance(d, dict):
                return {k: clean_meta(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_meta(i) for i in d]
            elif isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
                return None
            return d
        
        meta_clean = clean_meta(meta)
        
        resp = {"data": records, "meta": meta_clean}
        if meta_clean.get("error"):
            resp["error"] = meta_clean["error"]
        return jsonify(resp)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500


@socketio.on("connect")
def handle_connect():
    print(f"[WS] Client connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"[WS] Client disconnected: {sid}")

@socketio.on("join_symbol")
def handle_join_symbol(data):
    sid = request.sid
    symbol = data.get("symbol", "").upper()
    exchange = data.get("exchange", "").upper()  # e.g. "NSE", "MCX", "BSE"
    interval = int(data.get("interval", "15"))
    if not symbol:
        return

    # Leave all previous symbol rooms
    current_rooms = rooms(sid)
    for r in current_rooms:
        if r != sid:
            leave_room(r)

    join_room(symbol)
    print(f"[WS] Client {sid} joined room: {symbol}")
    
    # Wake up the scheduler if it was hibernating
    from dashboard.scheduler import wake_scheduler
    wake_scheduler()

    # ── Immediately serve the most recently available data on rejoin ──────────
    # After a long absence (e.g. phone switch tabs), the client missed data_updated
    # events. Instead of waiting up to 3 min for the next scheduler cycle, we
    # check the state RIGHT NOW and inform the client immediately.
    _notify_rejoining_client(sid, symbol, exchange, interval)


def _notify_rejoining_client(sid: str, symbol: str, exchange: str, interval: int = 15):
    """
    Immediately tell a rejoining client what data is available.

    Possible outcomes (emitted only to the rejoining client via `room=sid`):
      1. data_updated   — fresh data exists on disk → client re-fetches normally
      2. data_fetching  — a fetch is in progress right now → client shows spinner
      3. market_status  — market is closed / holiday / weekend → client shows message
    """
    from core.utils import get_last_trading_day
    from dashboard.scheduler import _fetch_locks, _locks_lock

    now = ist_now()
    today_str = now.strftime("%Y-%m-%d")

    # ── Resolve exchange if caller didn't send it ─────────────────────────────
    sym_upper = symbol.upper()
    if not exchange:
        if sym_upper in NSE_INDEX_KEYS or sym_upper in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
            exchange = "NSE"
        elif sym_upper in MCX_FUT_KEYS or sym_upper in {"CRUDEOIL", "NATURALGAS", "SILVER", "GOLD"}:
            exchange = "MCX"
        elif sym_upper in {"SENSEX", "BANKEX"}:
            exchange = "BSE"
        else:
            exchange = "NSE"  # safe default

    # ── Determine market hours for this exchange ──────────────────────────────
    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
    end_t   = datetime.strptime(cfg["end"],   "%H:%M:%S").time()
    now_t   = now.time()

    def secs(t):
        return t.hour * 3600 + t.minute * 60 + t.second

    trading_day = get_last_trading_day(now).strftime("%Y-%m-%d")
    is_trading_day = (trading_day == today_str)  # False on weekends

    market_open = is_trading_day and (secs(start_t) <= secs(now_t) <= secs(end_t))

    prefix   = "mcx" if exchange == "MCX" else "option"
    dir_name = "mcx_data" if exchange == "MCX" else "nse_data"
    sym_lc   = symbol.lower()
    csv_path  = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym_lc}_tabular_{trading_day}_int{interval}.csv")
    meta_path = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym_lc}_meta_{trading_day}_int{interval}.json")

    file_exists = os.path.exists(csv_path)
    file_age_s  = (time.time() - os.path.getmtime(csv_path)) if file_exists else None

    # ── Check if a fetch is currently running for this symbol ─────────────────
    # IMPORTANT: The acquire() must happen INSIDE _locks_lock so that concurrent
    # rejoining clients all observe the scheduler lock atomically. Without this,
    # Client A acquires the lock to check, and Clients B/C (racing concurrently)
    # see the lock held by A and incorrectly think a fetch is in progress.
    fetch_in_progress = False
    with _locks_lock:
        lock = _fetch_locks.get(symbol)
        if lock is not None:
            if lock.acquire(blocking=False):
                # Lock was free → no fetch in progress; release it right away
                lock.release()
                fetch_in_progress = False
            else:
                # Lock is held by the scheduler → a real fetch is running
                fetch_in_progress = True

    print(f"[WS] Rejoin check for {sid}/{symbol}: "
          f"market_open={market_open}, is_trading_day={is_trading_day}, "
          f"file_exists={file_exists}, age={file_age_s:.0f}s" if file_age_s is not None
          else f"[WS] Rejoin check for {sid}/{symbol}: "
               f"market_open={market_open}, is_trading_day={is_trading_day}, "
               f"file_exists={file_exists}, age=N/A")

    # ── Decision tree ─────────────────────────────────────────────────────────
    if fetch_in_progress:
        # A scheduler/API fetch is already running → tell the client to wait
        socketio.emit("data_fetching",
                      {"symbol": symbol, "exchange": exchange,
                       "message": f"Fetching data for {symbol} ({trading_day})…"},
                      room=sid)
        print(f"[WS] Told {sid} that fetch is in progress for {symbol} ({trading_day}).")
        return

    if file_exists:
        # Fresh data is on disk — send data_updated so the client re-polls /api/option-data
        socketio.emit("data_updated",
                      {"symbol": symbol, "prefix": exchange,
                       "timestamp": datetime.fromtimestamp(
                           os.path.getmtime(csv_path)).isoformat()},
                      room=sid)
        print(f"[WS] Sent data_updated to {sid} for {symbol} "
              f"(file age {file_age_s:.0f}s).")
        return

    # No file yet — explain why
    if not is_trading_day:
        # Weekend
        socketio.emit("market_status",
                      {"symbol": symbol, "exchange": exchange,
                       "status": "weekend",
                       "message": f"Markets are closed today (weekend). "
                                   f"Last trading day: {trading_day}."},
                      room=sid)
        print(f"[WS] Told {sid} market is closed (weekend) for {symbol}.")
    elif not market_open:
        # Weekday but outside trading hours (pre-market or post-market)
        market_end_str = cfg["end"][:5]
        market_start_str = cfg["start"][:5]
        if secs(now_t) < secs(start_t):
            message = (f"Market opens at {market_start_str} IST. "
                       f"Historical data will load once trading begins.")
            status = "pre_market"
        else:
            message = (f"Market closed at {market_end_str} IST. "
                       f"No data file found for {symbol} today ({trading_day}).")
            status = "post_market"
        socketio.emit("market_status",
                      {"symbol": symbol, "exchange": exchange,
                       "status": status, "message": message},
                      room=sid)
        print(f"[WS] Told {sid} market_status={status} for {symbol}.")
    else:
        # Market is open but data hasn't been fetched yet — it may be a holiday
        # or the very first fetch of the day is pending. Inform the client.
        socketio.emit("market_status",
                      {"symbol": symbol, "exchange": exchange,
                       "status": "fetching_initial",
                       "message": f"Market is open but initial data for {symbol} "
                                   f"is being fetched. Please wait…"},
                      room=sid)
        print(f"[WS] Told {sid} initial fetch pending for {symbol}.")


def get_active_symbols():
    """Returns a list of symbols currently being watched by at least one client."""
    # This is a bit of a hack since SocketIO doesn't expose a global room list easily
    # We'll use the rooms in the server's manager
    try:
        all_rooms = socketio.server.manager.rooms.get("/", {})
        # Filter for rooms that aren't private session IDs (usually session IDs are 32 chars)
        active = [r for r, sids in all_rooms.items() if len(sids) > 0 and len(str(r)) < 20]
        return active
    except Exception:
        return []
