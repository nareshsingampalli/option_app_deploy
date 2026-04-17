
import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

import upstox_client
from core.config import UPSTOX_ACCESS_TOKEN
from core.utils import ist_now

def test_mcx_fetch():
    print(f"Testing MCX fetch for today: {ist_now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)
    
    # Real instrument key for CRUDEOIL FUT
    spot_key = "MCX_FO|486502" 
    
    history_api = upstox_client.HistoryApi(api_client)
    
    try:
        # Test with "1minute" and api_version "2.0"
        print(f"Fetching intraday candles for {spot_key} with interval '1minute' and api_version '2.0'...")
        resp = history_api.get_intra_day_candle_data(spot_key, "1minute", "2.0")
        if resp.data and resp.data.candles:
            print(f"Success! Found {len(resp.data.candles)} candles.")
            print(f"Latest candle: {resp.data.candles[0]}")
        else:
            print("No candles returned.")
            print(f"Response: {resp}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mcx_fetch()
