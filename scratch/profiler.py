import sys
import os
sys.path.append(os.getcwd())
import time
import pandas as pd
import numpy as np
from core.black_scholes import implied_volatility

# Simulate a typical session (1-min data, 375 records)
n = 375
df = pd.DataFrame({
    'ltp': np.random.uniform(50, 100, n),
    'oi': np.random.uniform(10000, 20000, n),
    'volume': np.random.uniform(100, 1000, n)
})
df.index = pd.date_range("2026-04-16 09:15", periods=n, freq="1min")
spot_p = 22500.0
strike = 22500.0
T = 0.01

print(f"Profiling IV calculation for {n} rows...")
start = time.time()
iv_list = []
for idx, row in df.iterrows():
    iv = implied_volatility(row["ltp"], spot_p, strike, T, 0.05, "CE")
    iv_list.append(iv)
end = time.time()
print(f"Time taken (Sequential Loop): {end - start:.4f} seconds")

# Calculate for 12 instruments
print(f"Estimated time for 12 instruments: {(end - start) * 12:.4f} seconds")
