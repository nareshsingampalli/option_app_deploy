import sys
import os
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from core.config import NSE_INDEX_KEYS

def test_via_fetcher():
    key = NSE_INDEX_KEYS["NIFTY"]
    target_date = "2026-04-16"   # We want 16-Apr, system must find baseline

    print(f"--- HistoricalCandleFetcher.get_spot_candles({target_date}) ---")
    print(f"Expected: expansion stops when response has >= 2 unique trading days\n")

    fetcher = HistoricalCandleFetcher(interval=1)  # 1-min, same as URL you tested

    df = fetcher.get_spot_candles(key, target_date)

    if df is not None and not df.empty:
        unique_dates = df.index.strftime("%Y-%m-%d").unique().tolist()
        print(f"Unique dates in response : {unique_dates}")
        print(f"Total candles            : {len(df)}")
        print(f"Confirmed from_date used : {fetcher._confirmed_from_date}")
        print(f"First candle             : {df.index[0]}")
        print(f"Last candle              : {df.index[-1]}")
        print(f"Unique days count        : {len(unique_dates)}")
    else:
        print("EMPTY: No data returned.")

if __name__ == "__main__":
    test_via_fetcher()
