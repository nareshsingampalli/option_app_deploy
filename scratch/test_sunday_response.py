
import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path so we can import internal modules
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from fetchers.expired import ExpiredCandleFetcher
from core.config import NSE_INDEX_KEYS

def test_sunday_responses():
    print(f"=== Upstox Sunday/Weekend Response Test ===")
    
    nifty_key = NSE_INDEX_KEYS["NIFTY"] # NSE_INDEX|Nifty 50
    historical = HistoricalCandleFetcher(interval=15)
    expired = ExpiredCandleFetcher(interval=15)

    test_dates = ["2026-04-12", "2026-04-05"] # Sundays

    for date_str in test_dates:
        print(f"\n--- Testing Date: {date_str} (Sunday) ---")
        
        # 1. Historical API Check
        try:
            print(f"Checking Historical API for {date_str}...")
            df_hist = historical.fetch_single(nifty_key, "minutes", 15, date_str, date_str)
            print(f"Status: {historical.last_status}")
            print(f"Data Found: {'Yes' if df_hist is not None and not df_hist.empty else 'No (Empty)'}")
        except Exception as e:
            print(f"Historical Error: {e}")

        # 2. Expired API Check
        try:
            print(f"Checking Expired Data API for {date_str}...")
            # Note: For Expired, index keys often look like UNDERLYING|DD-MM-YYYY
            # But here we just want to see if the candle fetch returns 401, 200/empty, or 404
            df_exp = expired.fetch_candle_data(f"NIFTY|{date_str}", "15minute", date_str, date_str)
            print(f"Status: {expired.last_status}")
            print(f"Data Found: {'Yes' if df_exp is not None and not df_exp.empty else 'No (Empty)'}")
        except Exception as e:
            print(f"Expired Error: {e}")

if __name__ == "__main__":
    test_sunday_responses()
