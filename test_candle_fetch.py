# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
test_candle_fetch.py
====================
Fetches Nifty 50 option candle data for 2026-02-24
in both HISTORICAL and EXPIRED mode and prints a summary.

Usage:
    python test_candle_fetch.py
"""

import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
load_dotenv("/home/ubuntu/refactor_app/.env")

TARGET_DATE = "2026-02-24"
TARGET_TIME = None        # Set e.g. "09:15" for a time-specific snapshot
NUM_STRIKES = 2           # CE+PE pairs each side of ATM

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def summarise_df(df, label, max_rows=5):
    if df is None or df.empty:
        print(f"  [{label}] NO DATA returned (None or empty DataFrame)")
        return
    print(f"  [{label}] OK {len(df)} rows | index: {df.index[0]} -> {df.index[-1]}")
    print(f"  Columns: {list(df.columns)}")
    print(df.head(max_rows).to_string())

# ─────────────────────────────────────────────────────────────
# 1. HISTORICAL MODE — uses HistoricalCandleFetcher
# ─────────────────────────────────────────────────────────────

print_section(f"HISTORICAL MODE — {TARGET_DATE}")

try:
    from option_chain import HistoricalStrategy, get_option_chain_instruments

    hist_strategy = HistoricalStrategy()

    # Step 1: Spot price
    spot = hist_strategy.get_spot_price(TARGET_DATE, TARGET_TIME)
    print(f"\n  Spot Price (Historical): {spot}")

    if spot:
        # Step 2: Instruments
        instruments, expiry, is_expired = get_option_chain_instruments(
            spot, num_strikes=NUM_STRIKES, reference_date=TARGET_DATE
        )
        print(f"  Expiry: {expiry} | is_expired: {is_expired}")
        print(f"  Instruments found: {len(instruments)}")
        for i in instruments:
            print(f"    → {i['symbol']} | strike={i['strike']} | type={i['type']}")

        # Step 3: Candle data for first CE and first PE
        for inst in instruments[:4]:   # first 2 CE/PE pairs
            from_date = TARGET_DATE
            to_date   = TARGET_DATE
            print(f"\n  Fetching candles: {inst['symbol']} ...")
            df = hist_strategy.get_candle_data(inst, from_date, to_date)
            summarise_df(df, inst['symbol'])
    else:
        print("  ⚠️  Could not fetch spot price. Skipping instruments.")

except Exception as e:
    print(f"  ❌ HISTORICAL MODE ERROR: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 2. EXPIRED MODE — uses ExpiredCandleFetcher
# ─────────────────────────────────────────────────────────────

print_section(f"EXPIRED MODE — {TARGET_DATE}")

try:
    from option_chain import ExpiredStrategy, get_expired_option_chain_instruments

    exp_strategy = ExpiredStrategy()

    # Step 1: Spot price (uses HistoricalCandleFetcher for index)
    spot = exp_strategy.get_spot_price(TARGET_DATE, TARGET_TIME)
    print(f"\n  Spot Price (Expired): {spot}")

    if spot:
        # Step 2: Expired instruments
        instruments, expiry = get_expired_option_chain_instruments(
            spot, num_strikes=NUM_STRIKES, reference_date=TARGET_DATE
        )
        print(f"  Expiry: {expiry}")
        print(f"  Instruments found: {len(instruments)}")
        for i in instruments:
            print(f"    → {i['symbol']} | strike={i['strike']} | type={i['type']}")

        # Step 3: Candle data for first few instruments
        for inst in instruments[:4]:
            from_date = TARGET_DATE
            to_date   = TARGET_DATE
            print(f"\n  Fetching candles: {inst['symbol']} ...")
            df = exp_strategy.get_candle_data(inst, from_date, to_date)
            summarise_df(df, inst['symbol'])
    else:
        print("  ⚠️  Could not fetch spot price. Skipping instruments.")

except Exception as e:
    print(f"  ❌ EXPIRED MODE ERROR: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 3. IV spot map check (5-min candles for IV calculation)
# ─────────────────────────────────────────────────────────────

print_section(f"IV SPOT MAP — {TARGET_DATE} (5-min Nifty candles)")

try:
    from option_chain import HistoricalStrategy, ExpiredStrategy

    print("\n  [Historical fetcher — IV spot data]")
    h = HistoricalStrategy()
    df_h = h.get_iv_spot_data(TARGET_DATE)
    summarise_df(df_h, "Hist IV spot (5min)")

    print("\n  [Expired fetcher — IV spot data]")
    e = ExpiredStrategy()
    df_e = e.get_iv_spot_data(TARGET_DATE)
    summarise_df(df_e, "Expired IV spot (5min)")

except Exception as ex:
    print(f"  ❌ IV SPOT MAP ERROR: {ex}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60 + "\n")
