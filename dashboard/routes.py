"""Dashboard routes — all Flask API endpoints."""

from __future__ import annotations

import json
import os
import time
import threading
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from flask import jsonify, render_template, request
from flask_socketio import join_room, leave_room, rooms

from dashboard import app, socketio
from core.utils import ist_now, get_last_trading_day
from core.config import SCHEDULER_HOURS, CANDLE_INTERVAL_MINUTES, NSE_INDEX_KEYS, MCX_FUT_KEYS


# Global tracker for active client interests (sid -> {symbol, interval})
_active_clients = {}
_clients_lock = threading.Lock()


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


def _is_market_open(exchange: str, now=None) -> bool:
    """Returns True only if current IST time is within the exchange's trading window."""
    if now is None:
        now = ist_now()
    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
    end_t   = datetime.strptime(cfg["end"],   "%H:%M:%S").time()
    def s(t): return t.hour * 3600 + t.minute * 60 + t.second
    is_weekday = now.weekday() < 5
    return is_weekday and (s(start_t) <= s(now.time()) <= s(end_t))


def _is_data_stale(csv_path, exchange, interval):
    """
    Checks if data is stale, considering market hours.
    After market close, an existing file is NEVER stale — the session is
    complete and the broker has nothing new to serve.
    """
    if not os.path.exists(csv_path):
        return True

    now = ist_now()
    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    end_time = datetime.strptime(cfg["end"], "%H:%M:%S").replace(
        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
    )
    start_time = datetime.strptime(cfg["start"], "%H:%M:%S").replace(
        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
    )

    # ── After market close: file exists → data is complete, never stale ──────
    if now > end_time:
        print(f"[StaleCheck] Market closed. Treating existing file as complete: {os.path.basename(csv_path)}")
        return False

    # Before market open: nothing to fetch yet
    if now < start_time:
        return False

    # ── During market hours: check if we're missing a candle ─────────────────
    reference_time = min(now, end_time)
    elapsed_minutes = (reference_time - start_time).total_seconds() / 60
    intervals_passed = int(elapsed_minutes // interval)

    if intervals_passed == 0:
        return False

    expected_last_candle_time = start_time + timedelta(minutes=intervals_passed * interval)
    buffer_time = expected_last_candle_time + timedelta(seconds=60)

    if now > buffer_time:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(csv_path)).replace(tzinfo=now.tzinfo)
        print(f"[StaleCheck] {csv_path}: Checking if current file is complete...")
        if file_mtime < expected_last_candle_time:
            print(f"[StaleCheck] STALE: Modification time older than last expected candle.")
            return True

        try:
            with open(csv_path, 'rb') as f:
                f.seek(-2048, os.SEEK_END)
                lines = f.readlines()
                if lines:
                    last_ts_str = lines[-1].decode().split(',')[0].strip('"')
                    last_ts = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=now.tzinfo)
                    if last_ts < expected_last_candle_time - timedelta(minutes=interval):
                        return True
        except:
            pass

    return False


@app.route("/api/market-status")
def market_status():
    exchange = request.args.get("exchange", "NSE").upper()
    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    start_str = cfg["start"][:5]
    end_str   = cfg["end"][:5]

    now      = ist_now()
    today_str = now.strftime("%Y-%m-%d")
    now_t    = now.time()

    start_t  = datetime.strptime(cfg["start"], "%H:%M:%S").time()
    end_t    = datetime.strptime(cfg["end"],   "%H:%M:%S").time()

    is_trading_day = now.weekday() < 5 # Simplified for now

    def secs(t):
        return t.hour * 3600 + t.minute * 60 + t.second

    is_open = is_trading_day and (secs(start_t) <= secs(now_t) <= secs(end_t))
    is_pre_market = is_trading_day and secs(now_t) < secs(start_t)

    return jsonify({
        "exchange":      exchange,
        "is_open":       is_open,
        "is_pre_market": is_pre_market,
        "start":         start_str,
        "end":           end_str,
        "now_ist":       now.strftime("%H:%M:%S"),
    })


@app.route("/api/spot-probe")
def spot_probe():
    """
    Lightweight probe: fetches the live intraday spot candle for a symbol.
    Used by the client BEFORE turning on live mode to determine:
      - Is today a trading day (market open)?
      - Is today a holiday (spot data empty despite market hours)?
    Returns: { is_holiday: bool, spot_price: float|null, exchange: str, symbol: str }
    """
    exchange = request.args.get("exchange", "NSE").upper()
    symbol   = request.args.get("symbol",   "NIFTY").upper()

    # Only meaningful during market hours
    if not _is_market_open(exchange):
        return jsonify({"is_holiday": False, "spot_price": None, "reason": "market_closed"})

    try:
        from fetchers.intraday import IntradayCandleFetcher
        fetcher  = IntradayCandleFetcher()
        today    = ist_now().strftime("%Y-%m-%d")

        # Resolve spot key
        if exchange == "MCX":
            from resolvers.mcx_resolver import MCXInstrumentResolver
            spot_key = MCXInstrumentResolver().get_spot_key(symbol, today)
        elif exchange == "BSE":
            from core.config import BSE_INDEX_KEYS
            spot_key = BSE_INDEX_KEYS.get(symbol, "BSE_INDEX|SENSEX")
        else:
            spot_key = NSE_INDEX_KEYS.get(symbol, NSE_INDEX_KEYS["NIFTY"])

        df = fetcher.get_spot_candles(spot_key, today)

        if df is None or df.empty:
            print(f"[SpotProbe] {symbol} ({exchange}): empty -> holiday or pre-data")
            return jsonify({"is_holiday": True, "spot_price": None})

        spot_price = float(df["close"].iloc[-1])
        print(f"[SpotProbe] {symbol} ({exchange}): spot={spot_price}")
        return jsonify({"is_holiday": False, "spot_price": spot_price})

    except Exception as e:
        print(f"[SpotProbe] Error: {e}")
        return jsonify({"is_holiday": False, "spot_price": None, "error": str(e)}), 500


@app.route("/api/pre-market-status")
def pre_market_status():
    """
    Called on page load (before market opens).
    Returns what date & mode the client should use:
      - If today is a trading day but market hasn't opened yet → use previous trading day (historical)
      - If today is weekend/holiday → use previous trading day (historical)
    Auto-rolls back through holidays by trying up to 5 previous weekdays.
    """
    exchange = request.args.get("exchange", "NSE").upper()
    from core.utils import get_prev_trading_day

    now = ist_now()
    today_str = now.strftime("%Y-%m-%d")

    # Determine if today has any chance of data
    is_weekday = now.weekday() < 5
    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    start_t  = datetime.strptime(cfg["start"], "%H:%M:%S").time()
    end_t    = datetime.strptime(cfg["end"],   "%H:%M:%S").time()

    def secs(t): return t.hour * 3600 + t.minute * 60 + t.second

    is_open      = is_weekday and (secs(start_t) <= secs(now.time()) <= secs(end_t))
    is_pre_mkt   = is_weekday and secs(now.time()) < secs(start_t)
    after_close  = is_weekday and secs(now.time()) > secs(end_t)

    # During market hours → let the existing live logic handle it
    if is_open:
        return jsonify({"use_historical": False, "date": today_str, "reason": "market_open"})

    # Before market open or weekend → historical mode, previous trading day.
    # We probe Upstox directly to find the true last session with data.
    from fetchers.historical import HistoricalCandleFetcher
    from fetchers.intraday import IntradayCandleFetcher
    # Always probe with 1-min for maximum detection speed and early archival discovery
    hist_fetcher = HistoricalCandleFetcher(interval=1)
    live_fetcher = IntradayCandleFetcher(interval=1)

    
    candidate = today_str if after_close else get_prev_trading_day(today_str)
    print(f"[PreMarket] after_close={after_close}, starting probe with candidate={candidate}")
    
    # Resolve spot key for probe
    from core.config import NSE_INDEX_KEYS
    spot_key = NSE_INDEX_KEYS["NIFTY"]
    if exchange == "MCX":
        from resolvers.mcx_resolver import MCXInstrumentResolver
        spot_key = MCXInstrumentResolver().get_spot_key("CRUDEOIL", candidate)
    elif exchange == "BSE":
        from core.config import BSE_INDEX_KEYS
        spot_key = BSE_INDEX_KEYS.get("SENSEX", "BSE_INDEX|SENSEX")

    # Probe up to 7 days back to skip weekends/holidays
    probe_range = 7
    for i in range(probe_range):
        try:
            # 1. Primary: Try Historical (for most past dates)
            if candidate == today_str:
                # Post-market 'today': historical API is usually empty, use Intraday.
                df = live_fetcher.get_spot_candles(spot_key, candidate)
            else:
                df = hist_fetcher.get_spot_candles(spot_key, candidate)
            
            # 2. Bridge: During midnight maintenance (12am-1am), data for 'yesterday' 
            # might still be on the Intraday server and not yet on Historical.
            if (df is None or df.empty) and i == 0:
                 print(f"[PreMarket] {candidate} not on Historical yet. Probing Intraday latest...")
                 # We probe without a date to see what the live cache currently holds
                 df = live_fetcher.get_spot_candles(spot_key, None)
                 if df is not None and not df.empty:
                     # Check if the intraday data actually belongs to our target 'yesterday'
                     last_date = df.index[-1].strftime("%Y-%m-%d")
                     if last_date == candidate:
                         print(f"[PreMarket] Found {candidate} in Intraday live cache. Bridging...")
                     else:
                         print(f"[PreMarket] Intraday cache holds {last_date}, not {candidate}. Rejecting bridge.")
                         df = None

            if df is not None and not df.empty:

                print(f"[PreMarket] Found data session: {candidate}")
                break
            
            print(f"[PreMarket] {candidate} is empty. Rolling back...")
        except Exception as e:
            print(f"[PreMarket] Probe error for {candidate}: {e}")
            pass
        candidate = get_prev_trading_day(candidate)
    
    reason = "pre_market" if is_pre_mkt else ("after_close" if after_close else "weekend")
    return jsonify({"use_historical": True, "date": candidate, "reason": reason})


@app.route("/api/option-data")

def get_option_data():
    t_req = time.time()
    exchange  = request.args.get("exchange", "NSE").upper()
    symbol    = request.args.get("symbol", "NIFTY").upper()
    date_str  = request.args.get("date", ist_now().strftime("%Y-%m-%d"))
    time_str  = request.args.get("time", "") 
    interval  = int(request.args.get("interval", CANDLE_INTERVAL_MINUTES))
    live_mode = request.args.get("live", "false").lower() == "true"
    next_expiry = request.args.get("next_expiry", "false").lower() == "true"

    print(f"[API] Request: {symbol} | {date_str} {time_str} | Live:{live_mode} | Int:{interval}")

    now_ist     = ist_now()
    today_str   = now_ist.strftime("%Y-%m-%d")
    is_today    = (date_str == today_str)
    prefix      = "mcx" if exchange == "MCX" else "option"

    # When live mode is ON, always target today and clear the time filter.
    if live_mode:
        date_str = today_str

    # For any date that is not Today, we want one consistent file for the whole day.
    # We ignore the specific time_str for filename generation to prevent redundant fetches.
    if date_str != today_str:
        time_str = ""
    else:
        # For today (Live), we also don't use time in filename as it's continuously updated.
        time_str = ""

    suffix       = f"_{time_str.replace(':', '')}" if time_str else ""
    if next_expiry: suffix += "_next"
    suffix       += f"_int{interval}"
    sym          = symbol.lower()
    dir_name     = "nse_data" if prefix == "option" else "mcx_data"
    csv_path     = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_tabular_{date_str}{suffix}.csv")
    meta_path    = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_meta_{date_str}{suffix}.json")

    # ── Fetch Trigger ────────────────────────────────────────────────────────
    market_currently_open = _is_market_open(exchange)
    needs_fetch = not os.path.exists(csv_path)
    # Stale check only when market is open AND user explicitly wants live data
    if not needs_fetch and is_today and market_currently_open and live_mode:
        needs_fetch = _is_data_stale(csv_path, exchange, interval)

    if needs_fetch:
        from dashboard.scheduler import get_lock
        lock = get_lock(symbol)
        if lock.acquire(blocking=False):
            fetch_msg = "Fetching latest data..." if live_mode else f"Loading data for {date_str}..."
            print(f"[API] {symbol} fetch starting in background ({fetch_msg})")
            socketio.emit("data_fetching", {"symbol": symbol, "message": fetch_msg}, room=symbol)

            # Capture loop variables for the thread closure
            _date_str   = date_str
            _time_str   = time_str
            _live_mode  = live_mode
            _is_today   = is_today
            _exchange   = exchange
            _symbol     = symbol
            _interval   = interval
            _next_exp   = next_expiry
            _csv_path   = csv_path
            _dir_name   = dir_name
            _prefix     = prefix
            _sym        = sym

            def bg_fetch_task(rollback_count=0):
                t_bg = time.time()
                try:
                    from storage.db_storage import build_storage_chain
                    from strategies.handlers import build_pipeline
                    from core.utils import get_prev_trading_day
                    storage  = build_storage_chain()
                    pipeline = build_pipeline(
                        _exchange, _date_str, _live_mode, _symbol,
                        storage, interval=_interval, next_expiry=_next_exp
                    )
                    if pipeline:
                        pipeline.run(_symbol, _date_str, _time_str)

                    # ── Maintenance Bridge: If 'yesterday' is empty on Historical ──
                    # During 12am-1am, Upstox might not have moved yesterday's data 
                    # from Intraday to Historical yet. We try the bridge here.
                    if not os.path.exists(_csv_path) and _date_str == get_prev_trading_day(today_str):
                        print(f"[API-BG] {_date_str} empty on Historical API. Probing Intraday bridge...")
                        bridge_pipeline = build_pipeline(
                            _exchange, _date_str, True, _symbol,
                            storage, interval=_interval, next_expiry=_next_exp
                        )
                        if bridge_pipeline:
                            bridge_pipeline.run(_symbol, _date_str, _time_str)

                    if not os.path.exists(_csv_path):
                        # ── TRUE Holiday / no-data ──
                        if _exchange == "MCX":
                             print(f"[API-BG] No MCX data for {_date_str}. Rollback disabled as requested.")
                             return

                        # roll back to previous trading day (NSE/BSE)
                        # Safety: Only rollback if we haven't reached the limit (max 3 days).
                        # Also check if it was a 401 error (don't loop if unauthorized).
                        if rollback_count >= 3:
                            print(f"[API-BG] Max rollbacks reached for {_symbol}. Stopping.")
                            socketio.emit("error", {"symbol": _symbol, "message": "Could not find data after 3 days of rollback."}, room=_symbol)
                            return

                        print(f"[API-BG] No data returned for {_date_str} ({_symbol}). "
                              f"Treating as holiday — rolling back ({rollback_count + 1}/3)...")
                        
                        prev_date    = get_prev_trading_day(_date_str)
                        prev_suffix  = ("_next" if _next_exp else "") + f"_int{_interval}"
                        prev_csv     = os.path.join(
                            os.getcwd(), _dir_name,
                            f"{_prefix}_{_sym}_tabular_{prev_date}{prev_suffix}.csv"
                        )
                        
                        # We don't recursively call bg_fetch_task automatically in a loop here.
                        # Instead, we notify the client of the holiday detection.
                        # The client can then decide to switch or we can do one more fetch.
                        socketio.emit("holiday_detected", {
                            "symbol":        _symbol,
                            "date":          _date_str,
                            "fallback_date": prev_date,
                        }, room=_symbol)
                    else:
                        socketio.emit("data_updated", {
                            "symbol":      _symbol, 
                            "date":        _date_str,
                            "interval":    _interval,
                            "next_expiry": _next_exp
                        }, room=_symbol)

                except Exception as e:
                    print(f"[API-BG] Error: {e}")
                    socketio.emit("error", {"symbol": _symbol, "message": f"Data process error: {str(e)}"}, room=_symbol)
                finally:
                    print(f"[API-BG] Task finished for {_symbol} in {time.time() - t_bg:.2f}s")
                    try: lock.release()
                    except: pass

            import threading
            threading.Thread(target=bg_fetch_task, daemon=True).start()
        else:
            print(f"[API] {symbol} fetch already in progress, skipping trigger.")

    try:
        if not os.path.exists(csv_path):
            return jsonify({"error": "Processing...", "status": "fetching"}), 200
        
        df = pd.read_csv(csv_path).fillna("")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f: meta = json.load(f)
        
        # Inject live status for today's session during market hours
        if is_today and market_currently_open:
            meta["live"] = True
            meta["is_today"] = True
        
        res = jsonify({"status": "success", "data": df.to_dict(orient="records"), "meta": meta, "date": date_str})
        print(f"[API] Response for {symbol}: 200 OK | ProcessTime: {time.time() - t_req:.2f}s")
        return res
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@socketio.on("join_symbol")
def handle_join_symbol(data):
    sid = request.sid
    symbol = data.get("symbol", "").upper()
    interval = int(data.get("interval", 15))
    next_expiry = bool(data.get("next_expiry", False))
    if not symbol: return

    with _clients_lock:
        _active_clients[sid] = {
            "symbol": symbol, 
            "interval": interval,
            "next_expiry": next_expiry
        }

    # Notify scheduler to wake up and check the new active interest
    from dashboard.scheduler import wake_scheduler
    wake_scheduler()
    # Leave existing room ONLY if it's different from the new requested symbol
    for r in rooms(sid):
        if r != sid and r != symbol:
            leave_room(r)
            print(f"[WS] Client {sid} left room: {r}")
    
    if symbol not in rooms(sid):
        join_room(symbol)
        print(f"[WS] Client {sid} joined room: {symbol} (interval: {interval}m, next: {next_expiry})")
    else:
        print(f"[WS] Client {sid} already in room: {symbol}. Syncing context.")
    # ── Stale Data Handshake ──────────────────────────────────────────────────
    # Check if the server already has data newer than what the client possesses.
    last_updated = data.get("last_updated")
    exchange = data.get("exchange", "NSE").upper()

    try:
        from core.utils import ist_now
        today = ist_now().strftime("%Y-%m-%d")
        prefix = "mcx" if exchange == "MCX" else "option"
        dir_name = "nse_data" if prefix == "option" else "mcx_data"
        suffix = ("_next" if next_expiry else "") + f"_int{interval}"
        meta_path = os.path.join(os.getcwd(), dir_name, f"{prefix}_{symbol.lower()}_meta_{today}{suffix}.json")

        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
                server_fetched_at = meta.get("fetched_at")
                
                # Logic: Trigger update if client has NO data, or client data is OLDER than server
                if not last_updated or (server_fetched_at and server_fetched_at > last_updated):
                    print(f"[WS] Client {sid} needs sync for {symbol} ({last_updated} < {server_fetched_at}). Triggering update.")
                    emit("data_updated", {"symbol": symbol, "interval": interval, "next_expiry": next_expiry}, room=sid)
    except Exception as e:
        print(f"[WS] Stale check error for {symbol}: {e}")


@socketio.on("leave_symbol")
def handle_leave_symbol(data):
    """Explicit leave — called when the client turns off Live Mode."""
    sid = request.sid
    symbol = data.get("symbol", "").upper()
    with _clients_lock:
        if sid in _active_clients:
            del _active_clients[sid]
    if symbol:
        leave_room(symbol)
        print(f"[WS] Client {sid} left room (live off): {symbol}")


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    with _clients_lock:
        if sid in _active_clients:
            del _active_clients[sid]
    print(f"[WS] Client {sid} disconnected — cleaned up rooms and active tracking.")


def get_active_symbols() -> dict[str, set[tuple[int, bool]]]:
    """
    Returns a dictionary mapping symbol -> set of (interval, next_expiry) tuples.
    Example: {'NIFTY': {(15, False), (15, True)}, 'BANKNIFTY': {(5, False)}}
    """
    with _clients_lock:
        active = {}
        for config in _active_clients.values():
            sym = config['symbol']
            track = (config['interval'], config['next_expiry'])
            if sym not in active:
                active[sym] = set()
            active[sym].add(track)
        return active
