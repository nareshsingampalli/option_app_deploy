
import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add parent directory to path so we can import internal modules
sys.path.append(os.getcwd())

import upstox_client
from core.config import UPSTOX_ACCESS_TOKEN, NSE_INDEX_KEYS

def get_raw_responses():
    cfg = upstox_client.Configuration()
    cfg.access_token = UPSTOX_ACCESS_TOKEN
    api_client = upstox_client.ApiClient(cfg)
    
    expired_api = upstox_client.ExpiredInstrumentApi(api_client)
    nifty_key = NSE_INDEX_KEYS["NIFTY"]

    print("\n=== RAW EXPIRED RESPONSE: 2026-04-05 (Sunday) ===")
    try:
        # Using a known valid instrument key from the instrument list for Nifty Options
        # (This avoids the format error)
        instrument_key = "NSE_FO|52733" 
        
        print(f"Attempting to fetch expired historicals for {instrument_key} on 2026-04-05...")
        # Correct SDK argument name is expired_instrument_key
        resp = expired_api.get_expired_historical_candle_data(
            expired_instrument_key=instrument_key,
            interval="15minute",
            to_date="2026-04-05",
            from_date="2026-04-05"
        )
        print(json.dumps(resp.to_dict() if hasattr(resp, "to_dict") else resp, indent=2))
    except Exception as e:
        print(f"Expired raw error: {e}")

if __name__ == "__main__":
    get_raw_responses()
