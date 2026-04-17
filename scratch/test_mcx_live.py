
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
    print(f"Using Token: {UPSTOX_ACCESS_TOKEN[:10]}...")
    
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)
    
    # CRUDEOIL key from config
    spot_key = "MCX_COM|294" 
    
    history_api = upstox_client.HistoryApi(api_client)
    
    try:
        # Get intraday candles (1-min)
        print(f"Fetching intraday candles for {spot_key}...")
        resp = history_api.get_intra_day_candle_data(spot_key, "minutes", 1)
        if resp.data and resp.data.candles:
            print(f"Success! Found {len(resp.data.candles)} candles.")
            print(f"Latest candle: {resp.data.candles[0]}")
        else:
            print("No candles returned (data field empty or None).")
            print(f"Response: {resp}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mcx_fetch()
