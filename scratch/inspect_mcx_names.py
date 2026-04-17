import gzip
import json
import pandas as pd
import requests

url = "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz"
print(f"Downloading {url}...")
resp = requests.get(url)
data = gzip.decompress(resp.content)
df = pd.DataFrame(json.loads(data))

# Look for CRUDEOIL options
commodity = "CRUDEOIL"
options = df[df['trading_symbol'].str.contains(commodity) & (df['instrument_type'].isin(['CE', 'PE']))]
print("\nALL AVAILABLE CRUDEOIL Options (first 100):")
print(options[['trading_symbol', 'expiry', 'strike_price']].head(100).to_string())
