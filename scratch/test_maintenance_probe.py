import sys
import os
# Ensure root dir is in path
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from fetchers.intraday import IntradayCandleFetcher
from core.config import NSE_INDEX_KEYS

def run_diagnostic():
    # Use NIFTY as the probe target
    symbol = "NIFTY"
    key = NSE_INDEX_KEYS["NIFTY"]
    target_date = "2026-04-16" # Yesterday

    print(f"--- 6 AM Diagnostic (Current Time: 06:10 AM) ---")
    print(f"Targeting: {symbol} | Date: {target_date}\n")

    # 1. Test Historical with 15-min
    print(f"[1] Probing Historical API (15-min) for {target_date}...")
    h_fetcher = HistoricalCandleFetcher(interval=15)
    try:
        h_df = h_fetcher.get_spot_candles(key, target_date)
        if h_df is not None and not h_df.empty:
            actual_dates = h_df.index.strftime("%Y-%m-%d").unique().tolist()
            print(f"   Dates found: {actual_dates}")
            print(f"   Last candle: {h_df.index[-1]}")
        else:
            print(f"   Returned EMPTY.")
    except Exception as e:
        print(f"   ERROR: {e}")

    # 2. Test Historical with 1-min FALLBACK
    print(f"\n[2] Probing Historical API (1-min fallback) for {target_date}...")
    try:
        # We manually fetch 1-min for the target date
        h_df_1 = h_fetcher.fetch_single(key, "minutes", 1, target_date, target_date)
        if h_df_1 is not None and not h_df_1.empty:
            actual_dates = h_df_1.index.strftime("%Y-%m-%d").unique().tolist()
            print(f"   SUCCESS! 1-min data found for: {actual_dates}")
            print(f"   Candle count: {len(h_df_1)}")
            print(f"   Last candle: {h_df_1.index[-1]}")
        else:
            print(f"   Returned EMPTY (Even with 1-min).")
    except Exception as e:
        print(f"   ERROR: {e}")

    # 3. Test Intraday Latest
    print(f"\n[3] Probing Intraday API (Latest Snapshot)...")
    i_fetcher = IntradayCandleFetcher(interval=15)
    try:
        i_df = i_fetcher.get_spot_candles(key, None)
        if i_df is not None and not i_df.empty:
            last_date = i_df.index[-1].strftime("%Y-%m-%d")
            print(f"   Latest live data date: {last_date}")
        else:
            print(f"   Returned EMPTY.")
    except Exception as e:
        print(f"   ERROR: {e}")

if __name__ == "__main__":
    run_diagnostic()
