
import os
import json
import pandas as pd
import subprocess
import time
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()
# Also try loading from refactor_app .env (common location on VM)
load_dotenv("/home/ubuntu/refactor_app/.env")

app = Flask(__name__)

@app.route('/')
def option_comparison():
    """Render the Option Chain Comparison Visualizer"""
    return render_template('option_comparison.html')

@app.route('/api/option-data')
def get_option_data():
    """Serve the generated option chain tabular data as JSON for a specific date"""
    # Get date from query parameter, default to 2026-02-20
    date_str = request.args.get('date', '2026-02-20')
    time_str = request.args.get('time', '') # Optional time HH:MM
    live_mode = request.args.get('live', 'false').lower() == 'true'
    
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
        filename = f"option_data_tabular_{date_str}_{clean_time}.csv"
        meta_filename = f"option_meta_{date_str}_{clean_time}.json"
    else:
        filename = f"option_data_tabular_{date_str}.csv"
        meta_filename = f"option_meta_{date_str}.json"
        
    csv_path = os.path.join(os.getcwd(), filename)
    
    # Check if data is stale for TODAY
    is_today = date_str == datetime.now().strftime('%Y-%m-%d')
    needs_fetch = False
    
    if not os.path.exists(csv_path):
        needs_fetch = True
    elif live_mode:
        # For live mode, refresh if older than 4 minutes (since frontend asks every 5)
        # Or just always refresh?
        # Let's refresh if > 1 minute old to be safe.
        file_mtime = os.path.getmtime(csv_path)
        if time.time() - file_mtime > 60:
            needs_fetch = True
    elif is_today and not time_str: # Only auto-refresh if no specific static time is set
        # If file exists but is for today, check age
        file_mtime = os.path.getmtime(csv_path)
        # Refresh if older than 60 seconds
        if time.time() - file_mtime > 60:
            print(f"Data for today is stale ({int(time.time() - file_mtime)}s old). Refreshing...")
            needs_fetch = True

    if needs_fetch:
        # Trigger fetch if not found or stale
        # Note: fetching takes time, this might timeout the request if it takes > 30s
        # For better UX, this should be async, but for now we'll do blocking
        try:
            print(f"Fetching data for {date_str} {time_str} (Live: {live_mode})...")
            # Run the fetcher script with the date
            # Ensure python path is correct
            python_exe = sys.executable if 'sys' in globals() else 'python'
            
            cmd = [python_exe, "option_chain.py", date_str]
            if live_mode:
                cmd.append("--live")
            elif time_str:
                cmd.append(time_str)
                
            subprocess.run(cmd, check=True, timeout=300)
            
            # Check again
            if not os.path.exists(csv_path):
                 # Try to load meta even if csv doesn't exist (e.g. market closed, no data yet)
                 meta_path = os.path.join(os.getcwd(), meta_filename)
                 if os.path.exists(meta_path):
                     with open(meta_path, 'r') as f:
                         meta = json.load(f)
                     
                     if meta.get("expired_contracts"):
                         return jsonify({
                             "error": f"Contracts for {date_str} have expired and are not available.",
                             "meta": meta
                         }), 200 # Return 200 so frontend can handle meta display
                         
                     return jsonify({
                         "data": [],
                         "meta": meta
                     })
                     
                 return jsonify({"error": f"No data available for {date_str}. Market might be closed or data missing."}), 404
                 
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Data fetch script failed. Check server logs."}), 500
        except Exception as e:
            return jsonify({"error": f"Error fetching data: {str(e)}"}), 500

    try:
        df = pd.read_csv(csv_path)
        
        # Load metadata if available
        meta_path = os.path.join(os.getcwd(), meta_filename)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
        if meta.get("expired_contracts"):
             return jsonify({
                 "error": f"Contracts for {date_str} have expired and are not available.",
                 "meta": meta
             })

        return jsonify({
            "data": df.to_dict(orient='records'),
            "meta": meta
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

if __name__ == '__main__':
    import sys # Import here to use in the route if needed
    port = int(os.getenv("OPTION_DASHBOARD_PORT", 8002))
    print(f"Starting Option Chain Dashboard on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
