
import os
import json
import sys
import threading
import pandas as pd
import subprocess
import time
from flask import Flask, render_template, jsonify, request
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()
load_dotenv("/home/ubuntu/refactor_app/.env")

app = Flask(__name__)

# Per-symbol fetch locks — prevents concurrent fetches for the same symbol
_fetch_locks = {
    'NSE': threading.Lock(),
    'MCX': threading.Lock(),
}

# ------------------------------------------------------------------
# Backend Scheduler
# Runs fetch scripts every FETCH_INTERVAL_SECONDS during market hours.
# Frontend is stateless — it just reads precomputed files.
# ------------------------------------------------------------------
FETCH_INTERVAL_SECONDS = 330  # 5 minutes 30 seconds

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
    except Exception as e:
        print(f"[Scheduler-{symbol}] Fetch error: {e}")
    finally:
        lock.release()

def _symbol_scheduler(symbol, start_s, end_s, script):
    """
    Independent scheduler for one symbol.
    - Waits until start_s on startup (or next day if already past end_s).
    - Fetches every FETCH_INTERVAL_SECONDS (5m 30s) within market hours.
    - Sleeps until next day's start_s once market closes.
    """
    from datetime import timedelta
    start_t = datetime.strptime(start_s, '%H:%M:%S').time()
    end_t   = datetime.strptime(end_s,   '%H:%M:%S').time()
    print(f"[Scheduler-{symbol}] Started. Market hours: {start_s} - {end_s} IST | Interval: {FETCH_INTERVAL_SECONDS}s")

    while True:
        now = datetime.now()
        cur_secs   = _secs(now.time())
        start_secs = _secs(start_t)
        end_secs   = _secs(end_t)

        if cur_secs < start_secs:
            # Before market open — wait until start time today
            wait = start_secs - cur_secs
            print(f"[Scheduler-{symbol}] Pre-market. Sleeping {wait}s until {start_s}...")
            time.sleep(wait)

        elif cur_secs > end_secs:
            # After market close — sleep until start_s tomorrow
            tomorrow_start = datetime.combine(now.date() + timedelta(days=1), start_t)
            wait = (tomorrow_start - now).total_seconds()
            print(f"[Scheduler-{symbol}] Market closed. Sleeping {wait:.0f}s until tomorrow {start_s}...")
            time.sleep(max(1, wait))

        else:
            # In market hours — fetch then sleep for interval
            _run_fetch(symbol, script)
            time.sleep(FETCH_INTERVAL_SECONDS)


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
    
    if live_mode:
        # Scheduler handles live fetches — frontend just reads the latest file
        # Only trigger a one-time fetch if the file doesn't exist yet at all
        if not os.path.exists(csv_path):
            needs_fetch = True  # First run before scheduler has fired
    elif not os.path.exists(csv_path):
        needs_fetch = True  # Historical data not cached yet
    elif is_today and not time_str:
        file_mtime = os.path.getmtime(csv_path)
        if time.time() - file_mtime > 60:
            print(f"Data for today is stale ({int(time.time() - file_mtime)}s old). Refreshing...")
            needs_fetch = True

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
                
        if meta.get("expired_contracts"):
             return jsonify({
                 "error": f"Contracts for {date_str} have expired and are not available.",
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
    print(f"Starting independent schedulers (every {FETCH_INTERVAL_SECONDS}s during market hours):")

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

    app.run(host='0.0.0.0', port=8010, debug=False, threaded=True)
