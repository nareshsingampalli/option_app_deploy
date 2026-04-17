from fetchers.intraday import IntradayCandleFetcher
from resolvers.mcx_resolver import MCXInstrumentResolver
import pandas as pd

def test_mcx_future():
    fetcher = IntradayCandleFetcher(interval=15)
    resolver = MCXInstrumentResolver()
    
    date_str = "2026-04-17"
    for commodity in ["CRUDEOIL", "NATURALGAS"]:
        print(f"\n--- Testing Future data for {commodity} on {date_str} ---")
        
        # 1. Resolve Spot Key
        try:
            spot_key = resolver.get_spot_key(commodity, date_str)
            print(f"Resolved Spot Key: {spot_key}")
            
            # 2. Fetch Spot Candles
            df = fetcher.get_spot_candles(spot_key, date_str)
            
            if df is not None and not df.empty:
                print(f"Future Candles Found: {len(df)} total")
                print(df.tail(3).to_string())
            else:
                print(f"No Future candles found for {commodity}.")
        except Exception as e:
            print(f"Error test {commodity}: {e}")

if __name__ == "__main__":
    test_mcx_future()
