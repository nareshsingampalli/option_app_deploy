import sys
import os
sys.path.append(os.getcwd())

from fetchers.intraday import IntradayCandleFetcher
from core.config import NSE_INDEX_KEYS

def test_token_auth():
    print(f"--- Token Authorization Check (SDK v2) ---")
    fetcher = IntradayCandleFetcher()
    key = NSE_INDEX_KEYS["NIFTY"]
    
    try:
        # Upstox SDK v2 often needs api_version='2.0'
        resp = fetcher._quote_api.get_full_market_quote(key, api_version='2.0')
        if resp and resp.data:
            print("TOKEN IS VALID: Successfully received market quote.")
            ltp = resp.data[key].last_price
            print(f"Current {key} LTP: {ltp}")
        else:
            print("TOKEN ISSUE: API returned success but no data.")
    except Exception as e:
        print(f"TOKEN FAILED: {e}")

if __name__ == "__main__":
    test_token_auth()
