import pandas as pd
import gzip
import os

path = 'cache/bse_instruments.json.gz'
df = pd.read_json(path)
sensex_options = df[(df['name'] == 'SENSEX') & (df['instrument_type'].isin(['CE', 'PE']))]
print("Expiry column sample:")
print(sensex_options['expiry'].head().tolist())

# Try converting one
from datetime import datetime
val = sensex_options['expiry'].iloc[0]
print(f"Typical value: {val}")
if val > 1e10: # Likely ms
    print(f"Converted from ms: {datetime.fromtimestamp(val/1000)}")
else:
    print(f"Converted from s: {datetime.fromtimestamp(val)}")
