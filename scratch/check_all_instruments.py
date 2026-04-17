import sys
import os
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from fetchers.intraday import IntradayCandleFetcher
from core.config import NSE_INDEX_KEYS

def run_comprehensive_check():
    target_date = "2026-04-16"
    print(f"--- Comprehensive Diagnostic (6:22 AM) ---")
    print(f"Target: {target_date}\n")

    h_fetcher = HistoricalCandleFetcher(interval=15)
    i_fetcher = IntradayCandleFetcher(interval=15)

    # Testing NIFTY 50 and NIFTY BANK
    instruments = {
        "NIFTY": "NSE_INDEX|Nifty 50",
        "BANKNIFTY": "NSE_INDEX|Nifty Bank"
    }

    for name, key in instruments.items():
        print(f"CHECKING: {name} ({key})")
        
        # 1. Historical Check
        try:
            h_df = h_fetcher.get_spot_candles(key, target_date)
            if h_df is not None and not h_df.empty:
                actual_dates = h_df.index.strftime("%Y-%m-%d").unique().tolist()
                print(f"   Historical: Found dates {actual_dates}")
                print(f"   Last candle: {h_df.index[-1]}")
                if target_date in actual_dates:
                    print(f"   FOUND: {target_date} is successfully available.")
                else:
                    print(f"   STALE: Only contains up to {actual_dates[-1]}.")
            else:
                print("   Historical: Returned EMPTY.")
        except Exception as e:
            print(f"   Historical Error: {e}")

        # 2. Intraday Check (Live Snapshot)
        try:
            i_df = i_fetcher.get_spot_candles(key, None)
            if i_df is not None and not i_df.empty:
                last_date = i_df.index[-1].strftime("%Y-%m-%d")
                print(f"   Intraday (Live): Holding data for {last_date}")
                print(f"   Last candle: {i_df.index[-1]}")
            else:
                print("   Intraday (Live): EMPTY.")
        except Exception as e:
            print(f"   Intraday Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    run_comprehensive_check()
