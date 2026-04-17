
import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

import upstox_client
from core.config import UPSTOX_ACCESS_TOKEN
from core.utils import ist_now

def test_nse_fetch():
    print(f"Testing NSE fetch for today: {ist_now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)
    
    # NIFTY 50 key
    spot_key = "NSE_INDEX|Nifty 50" 
    
    history_api = upstox_client.HistoryApi(api_client)
    
    try:
        # Test with "minutes" and 1 (current code style)
        print(f"Fetching intraday candles for {spot_key} with unit='minutes' and interval=1...")
        resp = history_api.get_intra_day_candle_data(spot_key, "minutes", "1")
        if resp.data and resp.data.candles:
            print(f"Success! Found {len(resp.data.candles)} candles.")
        else:
            print("No candles returned.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_nse_fetch()
