import pandas as pd
import gzip
import os

path = 'cache/bse_instruments.json.gz'
if not os.path.exists(path):
    print("File not found")
else:
    df = pd.read_json(path)
    print("Columns:", df.columns.tolist())
    
    # Check for SENSEX entries
    sensex_df = df[df['name'].str.contains('SENSEX', na=False, case=False)]
    print("\nTotal SENSEX rows:", len(sensex_df))
    
    if not sensex_df.empty:
        print("\nUnique names:", sensex_df['name'].unique().tolist())
        print("Unique instrument types:", sensex_df['instrument_type'].unique().tolist())
        print("\nSample rows:")
        print(sensex_df[['instrument_key', 'name', 'trading_symbol', 'instrument_type', 'expiry']].head(20).to_string())
    else:
        print("\nNo rows containing 'SENSEX' found.")
        print("\nFirst 10 index names in the file:")
        print(df[df['instrument_type'] == 'INDEX']['name'].unique()[:10])
