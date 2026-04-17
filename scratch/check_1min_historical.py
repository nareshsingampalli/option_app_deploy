import sys
import os
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from core.config import NSE_INDEX_KEYS

def run_1min_check():
    target_date = "2026-04-16"
    print(f"--- 1-Minute Historical Check (6:25 AM) ---")
    print(f"Target: {target_date}\n")

    # Use 1-minute interval specifically
    h_fetcher = HistoricalCandleFetcher(interval=1)

    instruments = {
        "NIFTY": "NSE_INDEX|Nifty 50",
        "BANKNIFTY": "NSE_INDEX|Nifty Bank"
    }

    for name, key in instruments.items():
        print(f"CHECKING: {name} ({key})")
        try:
            # We fetch exactly April 16 to April 16 (1-minute)
            df = h_fetcher.fetch_single(key, "minutes", 1, target_date, target_date)
            
            if df is not None and not df.empty:
                actual_dates = df.index.strftime("%Y-%m-%d").unique().tolist()
                print(f"   SUCCESS: Found {len(df)} candles for dates {actual_dates}")
                print(f"   Last candle: {df.index[-1]}")
            else:
                print("   EMPTY: No 1-minute data found for April 16.")
        except Exception as e:
            print(f"   Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    run_1min_check()
