
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from fetchers.historical import HistoricalCandleFetcher
from core.utils import get_last_trading_day
from resolvers.nse_resolver import NSEActiveResolver

def verify_option_intervals():
    fetcher = HistoricalCandleFetcher()
    yesterday = get_last_trading_day(datetime.now()).strftime("%Y-%m-%d")
    
    # Use a real liquid option contract for yesterday
    # We need a spot price to resolve instruments
    spot_key = "NSE_INDEX|Nifty 50"
    spot_df = fetcher.fetch_single(spot_key, "minutes", 1, yesterday, yesterday)
    if spot_df is None or spot_df.empty:
        print("Could not fetch spot price for yesterday.")
        return
    
    spot_price = float(spot_df.iloc[-1]['close'])
    
    resolver = NSEActiveResolver()
    # Resolve NEAR-ATM options for yesterday (2026-04-16)
    instruments, expiry_dt, is_fresh = resolver.resolve("NIFTY", spot_price, yesterday, num_strikes=1)
    
    if not instruments:
        print("Could not resolve instruments for yesterday.")
        return
    
    # Pick a CE instrument
    ce_instruments = [i for i in instruments if i.option_type == "CE"]
    test_inst = ce_instruments[0] 
    symbol = test_inst.key
    
    intervals = [1, 3, 5, 15]
    results = {}
    
    print(f"Comparing last candle of {yesterday} for {symbol} across intervals...")
    
    for interval in intervals:
        fetcher.interval = interval
        df = fetcher.fetch_single(symbol, "minutes", interval, yesterday, yesterday)
        if df is not None and not df.empty:
            last_row = df.iloc[-1]
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
    print("\nCross-Check Results:")
    if 1 in results and 15 in results:
        ref_oi = results[1]['oi']
        matches = True
        for interval, data in results.items():
            if data['oi'] != ref_oi:
                print(f"!!! OI VARIATION: {interval}m has {data['oi']} but 1m has {ref_oi}")
                matches = False
        
        if matches:
            print("SUCCESS: Open Interest is consistent across all intervals.")
        else:
            print("Notice: Final OI differs slightly between 1m and larger aggregated intervals (Broker side).")
    else:
        print("Insufficient data for full comparison.")

if __name__ == "__main__":
    verify_option_intervals()
