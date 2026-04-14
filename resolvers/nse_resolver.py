"""
NSEInstrumentResolver
---------------------
Active contracts  — downloads NSE.csv.gz, filters OPTIDX + symbol.
Expired contracts — uses ExpiredCandleFetcher REST API.

The resolver is exchange-aware; the strategy selects which one to use.
"""

from __future__ import annotations

import gzip
import io
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

from core.config import NSE_INDEX_KEYS, NSE_INSTRUMENT_URL
from core.exceptions import InstrumentResolutionError
from fetchers.expired import ExpiredCandleFetcher
from resolvers.base import Instrument, InstrumentResolver


class NSEActiveResolver(InstrumentResolver):
    """Resolves active NSE option contracts from the instruments CSV."""

    def resolve(
        self,
        symbol:         str,
        spot_price:     float,
        reference_date: str,
        num_strikes:    int = 3,
        expiry_offset:  int = 0,
    ) -> tuple[list[Instrument], Optional[datetime], bool]:
        print(f"[NSEActiveResolver] Fetching instruments for {symbol} @ spot={spot_price} (offset={expiry_offset})")
        from core.utils import get_instrument_df
        df = get_instrument_df(NSE_INSTRUMENT_URL, "NSE")

        # Filter: OPTIDX + symbol name
        opts = df[
            (df["instrument_type"] == "OPTIDX") &
            (df["name"] == symbol.upper())
        ].copy()
        opts = opts.rename(columns={"tradingsymbol": "trading_symbol", "strike": "strike_price"})

        if opts.empty:
            return [], None, False

        opts["expiry"] = pd.to_datetime(opts["expiry"])
        ref_dt = pd.to_datetime(reference_date)

        # Candidates >= reference_date
        expiries = sorted(opts["expiry"].unique())
        candidates = [e for e in expiries if e >= ref_dt]
        if len(candidates) <= expiry_offset:
            return [], None, True   # Not enough expiries in the future
        
        target_expiry = candidates[expiry_offset]

        expiry_df = opts[opts["expiry"] == target_expiry].copy()
        strikes   = sorted(expiry_df["strike_price"].unique())
        idx       = int(np.abs(np.array(strikes) - spot_price).argmin())
        selected  = strikes[max(0, idx - num_strikes): idx + num_strikes + 1]

        instruments: list[Instrument] = []
        for strike in selected:
            for opt_type in ("CE", "PE"):
                rows = expiry_df[
                    (expiry_df["strike_price"] == strike) &
                    (expiry_df["option_type"] == opt_type)
                ]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                instruments.append(Instrument(
                    key=row["instrument_key"],
                    symbol=row["trading_symbol"],
                    strike=float(strike),
                    option_type=opt_type,
                    expiry=target_expiry.to_pydatetime(),
                ))

        print(f"[NSEActiveResolver] {len(instruments)} instruments around ATM")
        return instruments, target_expiry.to_pydatetime(), False


class NSEExpiredResolver(InstrumentResolver):
    """Resolves expired NSE option contracts from the Upstox expired-instruments API."""

    def __init__(self):
        self._fetcher = ExpiredCandleFetcher()

    def resolve(
        self,
        symbol:         str,
        spot_price:     float,
        reference_date: str,
        num_strikes:    int = 3,
        expiry_offset:  int = 0,
    ) -> tuple[list[Instrument], Optional[datetime], bool]:
        underlying = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])
        print(f"[NSEExpiredResolver] symbol={symbol}, underlying={underlying}, spot={spot_price} (offset={expiry_offset})")

        expiries = self._fetcher.fetch_expiries(underlying)
        ref_dt          = datetime.strptime(reference_date, "%Y-%m-%d")
        target_expiry_s = None

        if not expiries:
            print(f"[NSEExpiredResolver] fetch_expiries returned empty. Using fallback to determine expiry date.")
            try:
                from resolvers.nse_resolver import NSEActiveResolver
                _, exp_dt, _ = NSEActiveResolver().resolve(symbol, spot_price, reference_date, num_strikes=0, expiry_offset=expiry_offset)
                if exp_dt:
                    target_expiry_s = exp_dt.strftime("%Y-%m-%d")
                    print(f"[NSEExpiredResolver] Fallback to ActiveResolver expiry: {target_expiry_s}")
            except Exception as e:
                print(f"[NSEExpiredResolver] Fallback error: {e}")
            
            if not target_expiry_s:
                target_expiry_s = reference_date
                print(f"[NSEExpiredResolver] Fallback to reference_date: {target_expiry_s}")
        else:
            candidates = [str(exp) for exp in sorted(expiries) if datetime.strptime(str(exp), "%Y-%m-%d").date() >= ref_dt.date()]
            if len(candidates) > expiry_offset:
                target_expiry_s = candidates[expiry_offset]
            else:
                # No next candidate in expired list — the "next" expiry is still active.
                # Delegate to NSEActiveResolver to get current active instruments (e.g. April 21).
                # is_fresh=True signals the pipeline to use HistoricalCandleFetcher for these instruments.
                print(f"[NSEExpiredResolver] No next expired candidate at offset={expiry_offset}. Delegating to ActiveResolver.")
                from resolvers.nse_resolver import NSEActiveResolver
                return NSEActiveResolver().resolve(symbol, spot_price, reference_date, num_strikes, expiry_offset=0)

        target_expiry_dt = datetime.strptime(target_expiry_s, "%Y-%m-%d")
        contracts        = self._fetcher.fetch_contracts(underlying, target_expiry_s)
        if not contracts:
            print(f"[NSEExpiredResolver] fetch_contracts returned no data for {target_expiry_s}.")
            return [], target_expiry_dt, True

        # Convert SDK objects to DataFrame robustly
        cdf = pd.DataFrame([c.to_dict() if hasattr(c, 'to_dict') else (c if isinstance(c, dict) else c.__dict__) for c in contracts])
        
        # Robust column mapping for field name variations (SDK specific)
        if "strike_price" not in cdf.columns and "strike" in cdf.columns:
            cdf["strike_price"] = cdf["strike"]
        
        if "trading_symbol" not in cdf.columns and "tradingsymbol" in cdf.columns:
            cdf["trading_symbol"] = cdf["tradingsymbol"]

        if "option_type" not in cdf.columns and "instrument_type" in cdf.columns:
            # Some SDK calls return CE/PE in instrument_type
            first_val = str(cdf["instrument_type"].iloc[0])
            if first_val in ("CE", "PE"):
                cdf["option_type"] = cdf["instrument_type"]
        
        # Ensure we have the basic columns
        if "strike_price" not in cdf.columns or "option_type" not in cdf.columns:
            print(f"[NSEExpiredResolver] ERROR: Missing columns: {cdf.columns.tolist()}")
            return [], target_expiry_dt, True

        cdf["strike"] = cdf["strike_price"].astype(float)
        cdf["type"]   = cdf["option_type"] # Use corrected field

        strikes  = sorted(cdf["strike"].unique())
        atm_idx  = int(np.abs(np.array(strikes) - spot_price).argmin())
        selected = strikes[max(0, atm_idx - num_strikes): atm_idx + num_strikes + 1]

        instruments: list[Instrument] = []
        for strike in selected:
            for opt_type in ("CE", "PE"):
                rows = cdf[(cdf["strike"] == strike) & (cdf["type"] == opt_type)]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                instruments.append(Instrument(
                    key=row["instrument_key"],
                    symbol=row["trading_symbol"],
                    strike=float(strike),
                    option_type=opt_type,
                    expiry=target_expiry_dt,
                ))

        print(f"[NSEExpiredResolver] {len(instruments)} expired instruments selected")
        return instruments, target_expiry_dt, True
