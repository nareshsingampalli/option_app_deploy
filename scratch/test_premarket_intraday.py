import sys
import os
sys.path.append(os.getcwd())

from fetchers.intraday import IntradayCandleFetcher
from core.config import NSE_INDEX_KEYS

def test_live_raw_now():
    print(f"--- Raw Intraday Feed Check (Current Time: 6:09 AM) ---")
    # We use the raw _fetch to see exactly what the broker sends
    fetcher = IntradayCandleFetcher(interval=1)
    key = NSE_INDEX_KEYS["NIFTY"]
    
    try:
        # Calling Upstox Intraday API (No date sent to broker)
        df = fetcher._fetch(key, "minutes", 1, date_str=None)
        
        if df is None or df.empty:
            print("RESULT: Intraday server returned EMPTY.")
            print("Conclusion: The live cache is currently blank (pre-market).")
        else:
            print(f"RESULT: Found {len(df)} candles in the 'Live' feed.")
            print(f"First timestamp: {df.index[0]}")
            print(f"Last timestamp:  {df.index[-1]}")
            
            unique_dates = df.index.strftime("%Y-%m-%d").unique().tolist()
            print(f"Dates present: {unique_dates}")
            
    except Exception as e:
        print(f"ERROR: API call failed: {e}")

if __name__ == "__main__":
    test_live_raw_now()
