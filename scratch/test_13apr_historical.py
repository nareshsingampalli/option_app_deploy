import sys
import os
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from core.config import NSE_INDEX_KEYS

def test_13_apr_historical():
    target_date = "2026-04-13"
    key = NSE_INDEX_KEYS["NIFTY"]

    print(f"--- Historical Mode: Testing {target_date} ---")
    print(f"Note: April 13 is NSE expiry day\n")

    fetcher = HistoricalCandleFetcher(interval=15)

    # Step 1: Spot candles (triggers from_date expansion probe)
    print(f"[1] get_spot_candles for {target_date}...")
    df = fetcher.get_spot_candles(key, target_date)

    if df is not None and not df.empty:
        unique_dates = df.index.strftime("%Y-%m-%d").unique().tolist()
        print(f"   Dates in response: {unique_dates}")
        print(f"   Candle count: {len(df)}")
        print(f"   Confirmed from_date used: {fetcher._confirmed_from_date}")
        print(f"   First candle: {df.index[0]}")
        print(f"   Last candle:  {df.index[-1]}")
    else:
        print(f"   EMPTY: No spot data for {target_date}")

    # Step 2: One option instrument (simulating pipeline)
    print(f"\n[2] get_candles for one option instrument (NIFTY2641323000CE)...")
    option_key = "NSE_FO|NIFTY2641323000CE"
    try:
        opt_df = fetcher.get_candles(option_key, target_date)
        if opt_df is not None and not opt_df.empty:
            unique_dates = opt_df.index.strftime("%Y-%m-%d").unique().tolist()
            print(f"   Dates in response: {unique_dates}")
            print(f"   Candle count: {len(opt_df)}")
            print(f"   Used same from_date: {fetcher._confirmed_from_date}")
        else:
            print(f"   EMPTY: No option data (may need ExpiredCandleFetcher for this instrument)")
    except Exception as e:
        print(f"   ERROR: {e}")

if __name__ == "__main__":
    test_13_apr_historical()
