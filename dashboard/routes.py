"""Dashboard routes — all Flask API endpoints."""

from __future__ import annotations

import json
import os
import sys
import subprocess
import time

import pandas as pd
from flask import jsonify, render_template, request
from dotenv import load_dotenv
from datetime import datetime
from flask_socketio import join_room, leave_room, rooms

from dashboard import app, socketio
from core.utils import ist_now


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
            return "<pre>" + "".join(f.readlines()[-20:]) + "</pre>"
    except Exception as e:
        return f"Error reading logs: {e}", 500


@app.route("/api/refresh-token", methods=["POST"])
def refresh_token():
    try:
        env_file = os.getenv("ENV_FILE", "/home/ubuntu/refactor_app/.env")
        load_dotenv(env_file, override=True)
        date_str  = datetime.now().strftime("%Y-%m-%d")
        meta_path = os.path.join(os.getcwd(), f"option_meta_{date_str}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "Invalid token" in meta.get("error", ""):
                meta.pop("error", None)
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
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

    print(f"[API] exch={exchange}, symbol={symbol}, date={date_str}, time={time_str}, live={live_mode}")

    is_today    = date_str == ist_now().strftime("%Y-%m-%d")
    prefix      = "mcx" if exchange == "MCX" else "option"

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
    sym          = symbol.lower()
    
    # Map prefix to directory name
    dir_name = "nse_data" if prefix == "option" else "mcx_data"
    
    csv_path     = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_tabular_{date_str}{suffix}.csv")
    meta_path    = os.path.join(os.getcwd(), dir_name, f"{prefix}_{sym}_meta_{date_str}{suffix}.json")

    file_exists  = os.path.exists(csv_path)
    file_age_s   = time.time() - os.path.getmtime(csv_path) if file_exists else 999999
    
    # Needs fetch if:
    # 1. File doesn't exist
    # 2. It's live mode and file is > 60s old (fast refresh)
    # 3. It's today and file is > 300s old (to refresh the Accumulating File)
    needs_fetch  = not file_exists 
    if file_exists:
        if live_mode:
            needs_fetch = file_age_s > 60
        elif is_today:
            # Refresh the main today file every 5 mins if scheduler isn't active
            needs_fetch = file_age_s > 300
        else:
            # Historical (Yesterday/etc) - check if it was a fallback fetch
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        m = json.load(f)
                    if m.get("is_fallback"):
                        needs_fetch = True
                        print(f"[API] {date_str} has fallback data. Attempting upgrade fetch.")
                except:
                    pass
            if not needs_fetch:
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
        print(f"[API] Data missing or stale for {date_str}. Triggering fetch...")
        lock     = _get_fetch_locks().get(symbol)
        if lock is None:
             # Make sure we use the same lock dict from scheduler
             from dashboard.scheduler import get_lock
             lock = get_lock(symbol)
             
        print(f"[API] Acquiring lock for {symbol}...")
        acquired = lock.acquire(timeout=310)
        if acquired:
            try:
                print(f"[API] Lock acquired. Checking freshness...")
                if os.path.exists(csv_path) and time.time() - os.path.getmtime(csv_path) < 60:
                    print("[API] Data already fresh, skipping fetch.")
                else:
                    from storage.db_storage import build_storage_chain
                    from strategies.handlers import build_pipeline
                    
                    print(f"[API] Building pipeline for {exchange}...")
                    storage  = build_storage_chain()
                    pipeline = build_pipeline(exchange, date_str, live_mode, symbol, storage)

                    if pipeline:
                        print(f"[API] Running pipeline...")
                        pipeline.run(symbol, date_str, time_str)
                    else:
                        print(f"[API] Pipeline resolution failed for {date_str}")
                    
                    # Notify only clients in the specific symbol room
                    socketio.emit("data_updated", 
                                  {"prefix": exchange, "symbol": symbol, "timestamp": datetime.now().isoformat()},
                                  to=symbol)
            except Exception as e:
                print(f"[API] Fetch error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Fetch error: {e}"}), 500
            finally:
                try:
                    lock.release()
                    print(f"[API] Lock released for {symbol}")
                except RuntimeError:
                    pass
        else:
            print(f"[API] Failed to acquire lock for {symbol} within 310s")

        if not os.path.exists(csv_path):
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("error"):
                    return jsonify({"error": meta["error"], "meta": meta}), 200
                if meta.get("expired_contracts"):
                    return jsonify({"error": f"Contracts for {date_str} have expired.", "meta": meta}), 200
                return jsonify({"data": [], "meta": meta}), 200
            msg = f"No data available for {date_str}."
            if is_today and not live_mode:
                msg += " Enable Live mode for today's data."
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
    if not symbol:
        return

    # Leave all previous symbol rooms
    current_rooms = rooms(sid)
    for r in current_rooms:
        if r != sid:
            leave_room(r)

    join_room(symbol)
    print(f"[WS] Client {sid} joined room: {symbol}")

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
