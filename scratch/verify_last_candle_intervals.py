
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from core.utils import get_last_trading_day

def verify_intervals():
    fetcher = HistoricalCandleFetcher()
    yesterday = get_last_trading_day(datetime.now()).strftime("%Y-%m-%d")
    symbol = "NSE_INDEX|Nifty 50"
    
    intervals = [1, 3, 5, 15]
    results = {}
    
    print(f"Comparing last candle of {yesterday} for {symbol} across intervals...")
    
    for interval in intervals:
        fetcher.interval = interval
        df = fetcher.fetch_single(symbol, "minutes", interval, yesterday, yesterday)
        if df is not None and not df.empty:
            last_row = df.iloc[-1]
            # Get the exact index time to be sure
            last_time = df.index[-1].strftime("%H:%M")
            results[interval] = {
                "time": last_time,
                "close": last_row['close'],
                "oi": last_row['open_interest']
            }
            print(f"Interval {interval}m: Last candle at {last_time} -> Close: {last_row['close']}, OI: {last_row['open_interest']}")
        else:
            print(f"Interval {interval}m: No data found.")

    # Cross-check
    print("\nCheck Results:")
    ref = results.get(15)
    if not ref:
        print("Reference 15m data missing.")
        return

    all_match = True
    for interval, data in results.items():
        if data['close'] != ref['close']:
            print(f"!!! DISCREPANCY in Close Price for {interval}m: {data['close']} vs {ref['close']} (15m)")
            all_match = False
        if data['oi'] != ref['oi']:
            print(f"!!! DISCREPANCY in OI for {interval}m: {data['oi']} vs {ref['oi']} (15m)")
            all_match = False
    
    if all_match:
        print("PASS: Last candle data is identical across all intervals.")
    else:
        print("FAIL: Data varies across intervals.")

if __name__ == "__main__":
    verify_intervals()
