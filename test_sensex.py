from core.config import BSE_INDEX_KEYS
from fetchers.intraday import IntradayCandleFetcher
import pandas as pd

try:
    print(f"Testing SENSEX spot fetch for key: {BSE_INDEX_KEYS['SENSEX']}")
    f = IntradayCandleFetcher()
    df = f.get_spot_candles(BSE_INDEX_KEYS['SENSEX'], '2026-03-16')
    if df is not None:
        print("Fetch successful!")
        print(df.tail())
    else:
        print("Fetch returned None")
except Exception as e:
    print(f"Error: {e}")
