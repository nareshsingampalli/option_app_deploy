
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from fetchers.intraday import IntradayCandleFetcher
from core.utils import ist_now

def verify_nifty_live():
    print(f"Verifying live fetch for Nifty 50 at {ist_now()}...")
    fetcher = IntradayCandleFetcher(interval=1)
    
    # Use the symbol sent by the user
    symbol = "NSE_INDEX|Nifty 50"
    today = ist_now().strftime("%Y-%m-%d")
    
    try:
        df = fetcher.get_spot_candles(symbol, today)
        if df is not None and not df.empty:
            print(f"Success! Fetched {len(df)} candles for {symbol}.")
            print("Last 3 candles:")
            print(df.tail(3))
        else:
            print(f"Failed to fetch data for {symbol} on {today}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_nifty_live()
