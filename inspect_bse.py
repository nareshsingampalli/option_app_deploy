import pandas as pd; df=pd.read_json('cache/bse_instruments.json.gz'); print(df[df.instrument_type=='INDEX'][['instrument_key','name']].to_string())
