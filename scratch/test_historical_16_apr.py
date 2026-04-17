import sys
import os
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from core.config import NSE_INDEX_KEYS

def test_historical_specific():
    target = "2026-04-16"
    key = NSE_INDEX_KEYS["NIFTY"]
    
    print(f"--- Historical Specific Check (Current Time: 6:11 AM) ---")
    print(f"Targeting: NIFTY | Requested Date: {target}\n")
    
    fetcher = HistoricalCandleFetcher(interval=15)
    # Note: get_spot_candles fetches from (target-1) to (target)
    try:
        df = fetcher.get_spot_candles(key, target)
        
        if df is not None and not df.empty:
            actual_dates = df.index.strftime("%Y-%m-%d").unique().tolist()
            print(f"RESULT: API returned {len(df)} candles.")
            print(f"Actual dates found in data: {actual_dates}")
            print(f"Full timestamp of last candle: {df.index[-1]}")
            
            if target in actual_dates:
                print(f"VERIFIED: {target} data is now available.")
            else:
                print(f"STALE: The API returned data, but it stops at {actual_dates[-1]}. {target} is missing.")
        else:
            print("RESULT: Historical API returned EMPTY.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_historical_specific()
