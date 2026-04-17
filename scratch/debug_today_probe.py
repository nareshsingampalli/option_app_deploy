
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from fetchers.intraday import IntradayCandleFetcher
from core.utils import ist_now

def check_today():
    fetcher = IntradayCandleFetcher(interval=1)
    today = ist_now().strftime("%Y-%m-%d")
    symbol = "NSE_INDEX|Nifty 50"
    
    print(f"Checking intraday data for {symbol} on {today}...")
    df = fetcher.get_spot_candles(symbol, today)
    if df is not None and not df.empty:
        print(f"Success! Found {len(df)} candles.")
        print(f"Last candle: {df.index[-1]}")
    else:
        print("Empty or None returned.")

if __name__ == "__main__":
    check_today()
