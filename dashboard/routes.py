"""Dashboard routes — all Flask API endpoints."""

from __future__ import annotations

import json
import os
import sys
import subprocess
import time

import pandas as pd
import numpy as np
import shutil
from flask import jsonify, render_template, request
from datetime import datetime
from flask_socketio import join_room, leave_room, rooms

from dashboard import app, socketio
from core.utils import ist_now
from core.config import SCHEDULER_HOURS, NSE_INDEX_KEYS, MCX_FUT_KEYS, CANDLE_INTERVAL_MINUTES


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
    """
    from core.utils import get_last_trading_day

    exchange = request.args.get("exchange", "NSE").upper()
    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    start_str = cfg["start"][:5]   # "HH:MM"
    end_str   = cfg["end"][:5]     # "HH:MM"

    now      = ist_now()
    today_str = now.strftime("%Y-%m-%d")
    now_t    = now.time()

    start_t  = datetime.strptime(cfg["start"], "%H:%M:%S").time()
    end_t    = datetime.strptime(cfg["end"],   "%H:%M:%S").time()

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
        
        if core.config.reload_access_token():
            print(f"[API] Live config state verified.")
        
        for dr in ["nse_data", "mcx_data"]:
            meta_files = glob.glob(os.path.join(os.getcwd(), dr, "*meta*.json"))
            for path in meta_files:
                try:
                    with open(path) as f:
                        meta = json.load(f)
                    if "error" in meta and ("token" in meta["error"].lower() or "auth" in meta["error"].lower()):
                        meta.pop("error", None)
                        with open(path, "w") as f:
                            json.dump(meta, f, indent=4)
                except Exception: pass
                    
        return jsonify({"message": "Token refreshed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/option-data")
def get_option_data():
    """Main route used by the frontend to fetch tabular option data (spot + chain)."""
    exchange  = request.args.get("exchange", "NSE").upper()
    symbol    = request.args.get("symbol", "NIFTY").upper()
    date_str  = request.args.get("date", ist_now().strftime("%Y-%m-%d"))
    time_str  = request.args.get("time", "")  # HH:MM
    interval  = int(request.args.get("interval", CANDLE_INTERVAL_MINUTES))
    live_mode = request.args.get("live", "false").lower() == "true"
    next_expiry = request.args.get("next_expiry", "false").lower() == "true"

    is_today    = date_str == ist_now().strftime("%Y-%m-%d")
    prefix      = "mcx" if exchange == "MCX" else "option"

    if is_today and not live_mode:
        stat = market_status().get_json()
        if stat["is_open"]:
            live_mode = True
        else:
            print(f"[API] Date is today but market is CLOSED, proceeding with historical fetch.")

    requested_time = time_str if not live_mode else ""
    if live_mode:
        date_str = ist_now().strftime("%Y-%m-%d")
        time_str = ""

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    suffix       = f"_{time_str.replace(':', '')}" if time_str else ""
    if next_expiry:
        suffix += "_next"
    suffix       += f"_int{interval}"
    sym          = symbol.lower()
    dir_name     = "nse_data" if prefix == "option" else "mcx_data"
    csv_path     = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_tabular_{date_str}{suffix}.csv")
    meta_path    = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_meta_{date_str}{suffix}.json")

    file_exists  = os.path.exists(csv_path)
    needs_fetch  = not file_exists
    cached_meta  = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f: cached_meta = json.load(f)
        except: pass

    if file_exists:
        file_mtime     = os.path.getmtime(csv_path)
        meta_mtime     = os.path.getmtime(meta_path) if os.path.exists(meta_path) else 0
        last_check_age = time.time() - max(file_mtime, meta_mtime)
        
        if live_mode or is_today:
            now_dt = ist_now()
            m_end_h, m_end_m = (15, 40) if exchange != "MCX" else (23, 40)
            m_end_dt = now_dt.replace(hour=m_end_h, minute=m_end_m, second=0, microsecond=0)
            
            if now_dt > m_end_dt:
                needs_fetch = max(file_mtime, meta_mtime) < m_end_dt.timestamp()
            else:
                # ── Spot-Sensitive Live Refresh ──────────────────────────────
                # If spot moved > 0.15% since last resolution, we likely need new strikes
                spot_shifted = False
                if cached_meta.get("spot_price"):
                    try:
                        # Quickly peek at the latest spot from the existing CSV without full load
                        with open(csv_path, 'rb') as f:
                            f.seek(-1024, os.SEEK_END)
                            last_line = f.readlines()[-1].decode().split(',')
                            # spot_price is usually the last column or near it. 
                            # Better: just check against the cached_meta spot if we have a way to get 'now' spot.
                            # For now, rely on time-based + big moves observed in previous fetches.
                            pass 
                    except: pass

                # Reduce TTL to 60s for live mode to respond faster to spot changes
                needs_fetch = last_check_age > 60
        else:
            if cached_meta.get("is_fallback"): needs_fetch = last_check_age > 3600
            elif not cached_meta.get("has_data"): needs_fetch = last_check_age > 1800
            else: needs_fetch = False

    # Check for cached auth error
    if "Invalid token" in cached_meta.get("error", ""):
        return jsonify({"error": cached_meta["error"], "meta": cached_meta}), 200

    # ── High-Speed Main-File Bootstrapping Logic ──────────────────────────────
    if not file_exists and time_str:
        main_csv_path = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_tabular_{date_str}_int{interval}.csv")
        if os.path.exists(main_csv_path):
            try:
                mdf = pd.read_csv(main_csv_path)
                mdf["date"] = mdf["date"].astype(str)
                time_rows = mdf[mdf["date"].str.contains(fr"\s{time_str.replace(':', ':')}:")]
                if not time_rows.empty:
                    mdf["spot_price"] = pd.to_numeric(mdf["spot_price"], errors='coerce')
                    ref_spot = float(time_rows["spot_price"].iloc[-1])
                    avail_symbols = set(mdf["symbol"].unique())
                    from strategies.handlers import build_pipeline
                    from storage.db_storage import build_storage_chain
                    storage = build_storage_chain()
                    pipeline = build_pipeline(exchange, date_str, False, symbol, storage, interval=interval)
                    instruments, _, _ = pipeline.resolver.resolve(symbol, ref_spot, date_str)
                    needed_symbols = [inst.symbol for inst in instruments]
                    if all(s in avail_symbols for s in needed_symbols):
                        print(f"[API] Bootstrap successful for {time_str}.")
                        snap_df = mdf[mdf["date"].str.contains(fr"\s{time_str.replace(':', ':')}:")]
                        snap_df.to_csv(csv_path, index=False)
                        main_meta_path = main_csv_path.replace("tabular", "meta").replace(".csv", ".json")
                        if os.path.exists(main_meta_path):
                            import shutil
                            shutil.copy2(main_meta_path, meta_path)
                        file_exists = True
                        needs_fetch = False
            except Exception as e:
                print(f"[API] Bootstrap error: {e}")
                needs_fetch = True

    if needs_fetch:
        from dashboard.scheduler import get_lock
        lock = get_lock(symbol)
        if lock.acquire(blocking=False):
            socketio.emit("data_fetching", {"symbol": symbol, "message": f"Processing {date_str} data for {symbol}...\u2026"}, room=symbol)
            def bg_fetch_task():
                try:
                    from datetime import datetime, timedelta
                    curr_date = date_str
                    retries = 0
                    fetch_status = "success"
                    last_err_code = None

                    while retries < 5:
                        c_path = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_tabular_{curr_date}{suffix}.csv")
                        if not os.path.exists(c_path) or (time.time() - os.path.getmtime(c_path) > 60):
                            from storage.db_storage import build_storage_chain
                            from strategies.handlers import build_pipeline
                            storage = build_storage_chain()
                            pipeline = build_pipeline(exchange, curr_date, live_mode, symbol, storage, interval=interval, next_expiry=next_expiry)
                            if pipeline:
                                # Subscribe UI notification (Observer Pattern)
                                def on_instruments_resolved(instruments, d):
                                    # Create a serializable format for the UI
                                    ui_data = []
                                    for inst in instruments:
                                         base_sym = inst.symbol.split(' ')[0]
                                         ui_data.append({
                                             "symbol": inst.symbol,
                                             "strike": inst.strike,
                                             "type": inst.option_type,
                                             "label": f"{base_sym} {inst.strike} {inst.option_type}"
                                         })
                                    socketio.emit("instruments_changed", {"symbol": symbol, "instruments": ui_data}, room=symbol)
                                pipeline.subscribe_instruments(on_instruments_resolved)

                                pipeline.run(symbol, curr_date, time_str)
                                fetcher = getattr(pipeline, "fetcher", None)
                                if fetcher:
                                    last_err_code = getattr(fetcher, "last_error_code", None)
                                    if last_err_code == "UDAPI100050" or getattr(fetcher, "last_status", None) == 401:
                                        fetch_status = "auth_error"
                                    elif getattr(fetcher, "last_status", None) == 200 and not os.path.exists(c_path):
                                        fetch_status = "holiday"

                                if getattr(pipeline.fetcher, "last_status", None) == 404 or not os.path.exists(c_path) or os.path.getsize(c_path) == 0:
                                    from core.utils import get_last_trading_day
                                    dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=1)
                                    dt = get_last_trading_day(dt)
                                    curr_date = dt.strftime("%Y-%m-%d")
                                    retries += 1
                                    continue
                                
                                # Save status in metadata
                                m_path = c_path.replace("tabular", "meta").replace(".csv", ".json")
                                if os.path.exists(m_path):
                                    with open(m_path, "r") as f: m_data = json.load(f)
                                    m_data["fetch_status"] = fetch_status
                                    m_data["error_code"] = last_err_code
                                    with open(m_path, "w") as f: json.dump(m_data, f)

                                socketio.emit("data_updated", {
                                    "prefix": exchange, 
                                    "symbol": symbol, 
                                    "date": curr_date, 
                                    "status": fetch_status,
                                    "error_code": last_err_code,
                                    "timestamp": datetime.now().isoformat()
                                }, room=symbol)
                                break
                            else: break
                        else:
                            socketio.emit("data_updated", {"prefix": exchange, "symbol": symbol, "date": curr_date, "timestamp": datetime.now().isoformat()}, room=symbol)
                            break
                    if retries >= 5:
                        socketio.emit("market_status", {"symbol": symbol, "exchange": exchange, "status": "unavailable", "message": "Upstox data currently unavailable."}, room=symbol)
                except Exception as e: print(f"[API-BG] Error: {e}")

                finally:
                    try: lock.release()
                    except: pass
            import threading
            threading.Thread(target=bg_fetch_task, daemon=True).start()
        else:
            print(f"[API] Fetch in progress for {symbol}. Serving existing...")

        if not os.path.exists(csv_path):
            if os.path.exists(meta_path):
                with open(meta_path) as f: m = json.load(f)
                return jsonify({"error": m.get("error", "Processing..."), "meta": m}), 200
            return jsonify({"error": "Fetch in progress. Please wait..."}), 200

    try:
        df = pd.read_csv(csv_path)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f: meta = json.load(f)
        if requested_time and not df.empty:
            try:
                tgt = pd.to_datetime(f"{date_str} {requested_time}").tz_localize(None)
                df_dt = pd.to_datetime(df["date"], errors='coerce').dt.tz_localize(None)
                df = df[df_dt.notnull() & (df_dt <= tgt)].copy()
            except: pass
        if not df.empty and (not meta.get("spot_price") or requested_time):
            last_spot = df["spot_price"].iloc[-1]
            if pd.notnull(last_spot): meta["spot_price"] = float(last_spot)
        
        import numpy as np
        df_clean = df.fillna("")
        records = df_clean.to_dict(orient="records")
        def clean_meta_val(d):
            if isinstance(d, dict): return {k: clean_meta_val(v) for k, v in d.items()}
            elif isinstance(d, list): return [clean_meta_val(i) for i in d]
            elif isinstance(d, float) and (np.isnan(d) or np.isinf(d)): return None
            return d
        return jsonify({"status": meta.get("fetch_status", "success"), "data": records, "meta": clean_meta_val(meta)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@socketio.on("connect")
def handle_connect():
    print(f"[WS] Client connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    print(f"[WS] Client disconnected: {request.sid}")


@socketio.on("join_symbol")
def handle_join_symbol(data):
    sid = request.sid
    symbol = data.get("symbol", "").upper()
    exchange = data.get("exchange", "").upper()
    interval = int(data.get("interval", "15"))
    if not symbol: return
    for r in rooms(sid):
        if r != sid: leave_room(r)
    join_room(symbol)
    print(f"[WS] Client {sid} joined room: {symbol}")
    from dashboard.scheduler import wake_scheduler
    wake_scheduler()
    _notify_rejoining_client(sid, symbol, exchange, interval)


def _notify_rejoining_client(sid: str, symbol: str, exchange: str, interval: int = 15):
    from core.utils import get_last_trading_day
    from dashboard.scheduler import _fetch_locks, _locks_lock

    now = ist_now()
    today_str = now.strftime("%Y-%m-%d")
    trading_day = get_last_trading_day(now).strftime("%Y-%m-%d")
    is_trading_day = (trading_day == today_str)

    cfg = SCHEDULER_HOURS.get(exchange, SCHEDULER_HOURS["NSE"])
    start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
    end_t = datetime.strptime(cfg["end"], "%H:%M:%S").time()
    now_t = now.time()

    def secs(t): return t.hour * 3600 + t.minute * 60 + t.second
    market_open = is_trading_day and (secs(start_t) <= secs(now_t) <= secs(end_t))

    prefix = "mcx" if exchange == "MCX" else "option"
    dir_name = "mcx_data" if exchange == "MCX" else "nse_data"
    csv_path = os.path.join(os.getcwd(), dir_name, f"{prefix}_{symbol.lower()}_tabular_{trading_day}_int{interval}.csv")
    
    fetch_in_progress = False
    with _locks_lock:
        lock = _fetch_locks.get(symbol)
        if lock and not lock.acquire(blocking=False): fetch_in_progress = True
        elif lock: lock.release()

    if fetch_in_progress:
        socketio.emit("data_fetching", {"symbol": symbol, "exchange": exchange, "message": f"Fetching data for {symbol} ({trading_day})\u2026"}, room=sid)
        return

    if os.path.exists(csv_path):
        socketio.emit("data_updated", {"symbol": symbol, "prefix": exchange, "timestamp": datetime.fromtimestamp(os.path.getmtime(csv_path)).isoformat()}, room=sid)
        return

    if not is_trading_day:
        socketio.emit("market_status", {"symbol": symbol, "exchange": exchange, "status": "weekend", "message": f"Markets are closed today (weekend). Last trading day: {trading_day}."}, room=sid)
    elif not market_open:
        socketio.emit("market_status", {"symbol": symbol, "exchange": exchange, "status": "closed", "message": f"Market is closed. No data file found for {symbol} today ({trading_day})."}, room=sid)
    else:
        socketio.emit("market_status", {"symbol": symbol, "exchange": exchange, "status": "fetching_initial", "message": f"Market is open but initial data for {symbol} is being fetched. Please wait\u2026"}, room=sid)


def get_active_symbols():
    try:
        all_rooms = socketio.server.manager.rooms.get("/", {})
        return [r for r, sids in all_rooms.items() if len(sids) > 0 and len(str(r)) < 20]
    except: return []
