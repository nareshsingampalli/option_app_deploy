
import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

import upstox_client
from core.config import UPSTOX_ACCESS_TOKEN
from core.utils import ist_now

def test_v3_fetch():
    print(f"Testing V3 fetch for today: {ist_now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)
    
    # NIFTY 50 key
    spot_key = "NSE_INDEX|Nifty 50" 
    
    v3_api = upstox_client.HistoryV3Api(api_client)
    
    try:
        # V3 style: (key, unit='minutes', interval=1)
        print(f"Fetching intraday candles (V3) for {spot_key} with unit='minutes' and interval='1'...")
        # Note: interval_value must be a STRING in some versions of SDK, check signature again: 
        # (self, instrument_key, unit, interval, **kwargs)
        resp = v3_api.get_intra_day_candle_data(spot_key, "minutes", 1)
        if resp.data and resp.data.candles:
            print(f"V3 Success! Found {len(resp.data.candles)} candles.")
        else:
            print("V3: No candles returned.")
            
    except Exception as e:
        print(f"V3 Error: {e}")

if __name__ == "__main__":
    test_v3_fetch()
