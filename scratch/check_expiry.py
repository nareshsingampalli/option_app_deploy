import sys
sys.path.insert(0, '.')
from resolvers.nse_resolver import NSEActiveResolver
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')
resolver = NSEActiveResolver()

_, expiry_dt, _ = resolver.resolve('NIFTY', 0, today, num_strikes=0, expiry_offset=0)
_, next_expiry_dt, _ = resolver.resolve('NIFTY', 0, today, num_strikes=0, expiry_offset=1)

print(f"Today            : {today}")
print(f"Current Expiry   : {expiry_dt.strftime('%Y-%m-%d, %A') if expiry_dt else 'None'}")
print(f"Next Expiry      : {next_expiry_dt.strftime('%Y-%m-%d, %A') if next_expiry_dt else 'None'}")
