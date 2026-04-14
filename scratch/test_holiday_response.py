
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path so we can import internal modules
sys.path.append(os.getcwd())

from fetchers.intraday import IntradayCandleFetcher
from fetchers.historical import HistoricalCandleFetcher
from fetchers.expired import ExpiredCandleFetcher
from core.config import UPSTOX_ACCESS_TOKEN, NSE_INDEX_KEYS

def test_holiday_api_responses():
    print(f"=== Upstox Holiday Response Test (Date: {datetime.now().strftime('%Y-%m-%d')}) ===")
    print(f"Access Token: {UPSTOX_ACCESS_TOKEN[:10]}...{UPSTOX_ACCESS_TOKEN[-10:]}")
    
    target_date = datetime.now().strftime("%Y-%m-%d")
    nifty_key = NSE_INDEX_KEYS["NIFTY"] # NSE_INDEX|Nifty 50
    
    # 1. Intraday Mode
    print("\n--- 1. Testing IntradayCandleFetcher ---")
    intraday = IntradayCandleFetcher(interval=15)
    try:
        # _fetch is protected, but we can call get_candles or use the internal method
        df_intraday = intraday.get_candles(nifty_key, target_date)
        print(f"Intraday Status: {intraday.last_status}")
        if df_intraday is not None:
            print(f"Intraday Data Found: {len(df_intraday)} rows")
            print(df_intraday.head())
        else:
            print("Intraday Result: None (Expected for holiday)")
    except Exception as e:
        print(f"Intraday Error: {e}")

    # 2. Historical Mode
    print("\n--- 2. Testing HistoricalCandleFetcher ---")
    historical = HistoricalCandleFetcher(interval=15)
    try:
        # Fetch for today (holiday)
        df_hist = historical.fetch_single(nifty_key, "minutes", 15, target_date, target_date)
        print(f"Historical Status: {historical.last_status}")
        if df_hist is not None:
            print(f"Historical Data Found: {len(df_hist)} rows")
        else:
            print("Historical Result: None (Expected for holiday)")
    except Exception as e:
        print(f"Historical Error: {e}")

    # 3. Expiry Fetcher (Metadata/Status check)
    print("\n--- 3. Testing ExpiredCandleFetcher (Expiries List) ---")
    expired = ExpiredCandleFetcher()
    try:
        # Testing underlying expiries for NSE_INDEX|Nifty 50
        expiries = expired.fetch_expiries(nifty_key)
        if expiries:
            print(f"Expiries Found: {len(expiries)} items")
            print(f"First 3 expiries: {expiries[:3]}")
        else:
            print("Expiries Result: Empty list (Check if token is valid)")
    except Exception as e:
        print(f"Expiry fetch error: {e}")

if __name__ == "__main__":
    # Create scratch directory if not exists
    os.makedirs("scratch", exist_ok=True)
    test_holiday_api_responses()
