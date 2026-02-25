
import os
import json
import sys
import threading
import pandas as pd
import subprocess
import time
from flask import Flask, render_template, jsonify, request
from datetime import datetime
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# Load environment
load_dotenv()
load_dotenv("/home/ubuntu/refactor_app/.env")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@socketio.on('connect')
def handle_connect():
    print(f"[WebSocket] Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WebSocket] Client disconnected: {request.sid}")

# Per-symbol fetch locks — prevents concurrent fetches for the same symbol
_fetch_locks = {
    'NSE': threading.Lock(),
    'MCX': threading.Lock(),
}

# ------------------------------------------------------------------
# Backend Scheduler
# Runs fetch scripts at minute % 5 == 1 during market hours.
# Frontend is stateless — it just reads precomputed files.
# ------------------------------------------------------------------

MARKET_HOURS = {
    #  symbol  start      end       script                  prefix
    'NSE': ('09:15:20', '15:40:00', 'option_chain.py',     'option'),
    'MCX': ('09:15:20', '23:59:00', 'option_chain_mcx.py', 'mcx'),
}

def _secs(t):
    """Convert time object to total seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second

def _run_fetch(symbol, script):
    """Run the fetch script for a symbol, protected by its lock."""
    today = datetime.now().strftime('%Y-%m-%d')
    lock = _fetch_locks[symbol]
    if not lock.acquire(blocking=False):
        print(f"[Scheduler-{symbol}] Already running, skipping this cycle.")
        return
    try:
        print(f"[Scheduler-{symbol}] Fetching live data for {today}...")
        subprocess.run([sys.executable, "-u", script, today, "--live"], timeout=280)
        print(f"[Scheduler-{symbol}] Fetch complete.")
        # Broadcast update to all clients
        socketio.emit('data_updated', {'prefix': symbol, 'timestamp': datetime.now().isoformat()}, namespace='/')
    except Exception as e:
        print(f"[Scheduler-{symbol}] Fetch error: {e}")
    finally:
        lock.release()

def _symbol_scheduler(symbol, start_s, end_s, script):
    """
    Independent scheduler for one symbol.
    - Waits until start_s on startup (or next day if already past end_s).
    - Fetches at every minute % 5 == 1 (e.g. 09:16, 09:21) within market hours.
    - Sleeps until next day's start_s once market closes.
    """
    from datetime import timedelta
    start_t = datetime.strptime(start_s, '%H:%M:%S').time()
    end_t   = datetime.strptime(end_s,   '%H:%M:%S').time()
    print(f"[Scheduler-{symbol}] Started. Market hours: {start_s} - {end_s} IST | Mode: Minute % 5 == 1")

    last_fetch_min = -1
    while True:
        now = datetime.now()
        cur_secs   = _secs(now.time())
        start_secs = _secs(start_t)
        end_secs   = _secs(end_t)

        if cur_secs < start_secs:
            # Before market open — wait until start time today
            wait = start_secs - cur_secs
            print(f"[Scheduler-{symbol}] Pre-market. Sleeping {wait}s until {start_s}...")
            last_fetch_min = -1 # Reset for the new day
            time.sleep(wait)

        elif cur_secs > end_secs:
            # After market close — sleep until start_s tomorrow
            tomorrow_start = datetime.combine(now.date() + timedelta(days=1), start_t)
            wait = (tomorrow_start - now).total_seconds()
            print(f"[Scheduler-{symbol}] Market closed. Sleeping {wait:.0f}s until tomorrow {start_s}...")
            last_fetch_min = -1
            time.sleep(max(1, wait))

        else:
            # In market hours
            # ── 1. Startup period (09:16 - 09:20): fetch every minute ────────
            #    Market starts at 09:15. We wait until 09:16 for the first 1-min candle.
            is_startup = (now.hour == 9 and 16 <= now.minute <= 20)
            
            # ── 2. Normal period: fetch every 5 mins at minute % 5 == 1 ─────
            #    e.g. 09:16, 09:21, 09:26, etc.
            is_regular_cycle = (now.minute % 5 == 1)

            if (is_startup or is_regular_cycle) and now.minute != last_fetch_min:
                mode_label = "Startup (1min)" if is_startup else "Regular (5min)"
                print(f"[Scheduler-{symbol}] Triggering {mode_label} fetch at {now.strftime('%H:%M')}")
                _run_fetch(symbol, script)
                last_fetch_min = now.minute
            
            # Sleep for a short burst to re-check
            time.sleep(15)


@app.route('/')
def option_comparison():
    """Render the Option Chain Comparison Visualizer"""
    return render_template('option_comparison.html')

@app.route('/api/refresh-token', methods=['POST'])
def refresh_token():
    """Force reload of environment variables from refactor_app .env"""
    try:
        # Reload env from the specific path
        load_dotenv("/home/ubuntu/refactor_app/.env", override=True)
        
        # Clear any existing error metadata for today to allow new fetch
        date_str = datetime.now().strftime('%Y-%m-%d')
        meta_filename = f"option_meta_{date_str}.json"
        meta_path = os.path.join(os.getcwd(), meta_filename)
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                # If there's an auth error, remove it so auto-fetch can try again
                if meta.get("error") and "Invalid token" in meta.get("error", ""):
                    print("Clearing invalid token error from metadata...")
                    meta.pop("error", None)
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=4)
            except:
                pass
                
        return jsonify({"message": "Environment variables refreshed successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to refresh environment: {str(e)}"}), 500


@app.route('/api/option-data')
def get_option_data():
    """Serve the generated option chain tabular data as JSON for a specific date"""
    # Get params
    date_str = request.args.get('date', '2026-02-20')
    time_str = request.args.get('time', '') # Optional time HH:MM
    live_mode = request.args.get('live', 'false').lower() == 'true'
    symbol_mode = request.args.get('symbol', 'NSE').upper() 
    
    print(f"[API] Request: symbol={symbol_mode}, date={date_str}, time={time_str}, live={live_mode}")

    prefix = "mcx" if symbol_mode == "MCX" else "option"
    script_name = "option_chain_mcx.py" if symbol_mode == "MCX" else "option_chain.py"

    if live_mode:
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = '' # Ignore time slider in live mode

    # Validate date format YYYY-MM-DD
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    if time_str:
        clean_time = time_str.replace(":", "")
        filename = f"{prefix}_data_tabular_{date_str}_{clean_time}.csv"
        meta_filename = f"{prefix}_meta_{date_str}_{clean_time}.json"
    else:
        filename = f"{prefix}_data_tabular_{date_str}.csv"
        meta_filename = f"{prefix}_meta_{date_str}.json"
        
    csv_path = os.path.join(os.getcwd(), filename)
    
    is_today = date_str == datetime.now().strftime('%Y-%m-%d')
    needs_fetch = False
    
    # Check if the file exists and how fresh it is
    file_exists = os.path.exists(csv_path)
    file_age_s = time.time() - os.path.getmtime(csv_path) if file_exists else 999999
    
    if not file_exists:
        needs_fetch = True
    elif live_mode or is_today:
        # For today/live data, if it's older than 2 minutes, refresh it on-demand
        if file_age_s > 120:
            needs_fetch = True
    
    # If it's a specific past date (not today), and file exists, we don't need to re-fetch.
    if not needs_fetch:
         print(f"[API] Serving existing file: {filename} (Age: {int(file_age_s)}s)")

    # Check if we should skip fetch due to existing auth error in metadata
    meta_path = os.path.join(os.getcwd(), meta_filename)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                existing_meta = json.load(f)
            if existing_meta.get("error") and "Invalid token" in existing_meta.get("error", ""):
                print("Skipping auto-fetch due to existing invalid token error.")
                return jsonify({
                    "error": existing_meta["error"],
                    "meta": existing_meta
                }), 200
        except:
            pass

    if needs_fetch:
        lock = _fetch_locks.get(symbol_mode, threading.Lock())
        acquired = lock.acquire(timeout=310)
        if not acquired:
            print(f"Lock timeout for {symbol_mode}, serving existing data if any.")
        else:
            try:
                # Re-check after lock: another tab may have already fetched
                if os.path.exists(csv_path) and time.time() - os.path.getmtime(csv_path) < 60:
                    print(f"Data already fresh (fetched by another tab). Skipping fetch.")
                else:
                    print(f"Fetching {symbol_mode} data for {date_str} {time_str} (Live: {live_mode})...")
                    cmd = [sys.executable, "-u", script_name, date_str]
                    if live_mode:
                        cmd.append("--live")
                    elif time_str:
                        cmd.append(time_str)
                    subprocess.run(cmd, check=True, timeout=300)
            except subprocess.CalledProcessError:
                print("Data fetch script failed.")
                return jsonify({"error": "Data fetch script failed. Check server logs."}), 500
            except Exception as e:
                return jsonify({"error": f"Error fetching data: {str(e)}"}), 500
            finally:
                try:
                    lock.release()
                except RuntimeError:
                    pass  # already released

        # Post-fetch: if CSV still missing, check meta for reason
        if not os.path.exists(csv_path):
            meta_path = os.path.join(os.getcwd(), meta_filename)
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get("error"):
                    return jsonify({"error": meta["error"], "meta": meta}), 200
                if meta.get("expired_contracts"):
                    return jsonify({"error": f"Contracts for {date_str} have expired.", "meta": meta}), 200
                return jsonify({"data": [], "meta": meta})
            return jsonify({"error": f"No data available for {date_str}. Market might be closed."}), 404


    try:
        df = pd.read_csv(csv_path)
        
        # Load metadata if available
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        
        # Fallback: obtain spot price from file data if meta is missing it
        if not meta.get("spot_price") and not df.empty and "spot_price" in df.columns:
            try:
                if time_str:
                    # Try to find the nearest match for the requested time
                    df['date_dt'] = pd.to_datetime(df['date'])
                    target_dt = pd.to_datetime(f"{date_str} {time_str}")
                    # Find rows closest to target time
                    df['diff'] = (df['date_dt'] - target_dt).abs()
                    best_match = df.sort_values('diff').iloc[0]
                    meta["spot_price"] = float(best_match["spot_price"])
                    print(f"[API] Recovered spot price {meta['spot_price']} from CSV for {time_str}")
                else:
                    meta["spot_price"] = float(df["spot_price"].iloc[-1])
            except Exception as e:
                print(f"[API] Failed to recover spot price from CSV: {e}")
                
        if meta.get("expired_contracts") and not meta.get("has_data"):
             return jsonify({
                 "error": f"Archived contracts for {date_str} are not available.",
                 "meta": meta
             })

        response = {
            "data": df.to_dict(orient='records'),
            "meta": meta
        }
        if meta.get("error"):
            response["error"] = meta["error"]
            
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500


if __name__ == '__main__':

    print(f"Starting Option Chain Dashboard on port {8010}...")
    print(f"Starting independent schedulers (Mode: Minute % 5 == 1 during market hours):")

    # One independent thread per symbol — NSE and MCX never block each other
    for sym, (start_s, end_s, script, prefix) in MARKET_HOURS.items():
        t = threading.Thread(
            target=_symbol_scheduler,
            args=(sym, start_s, end_s, script),
            daemon=True,
            name=f"Scheduler-{sym}"
        )
        t.start()
        print(f"  [{sym}] {start_s} - {end_s} IST -> {script}")

    socketio.run(app, host='0.0.0.0', port=8010, debug=False)
