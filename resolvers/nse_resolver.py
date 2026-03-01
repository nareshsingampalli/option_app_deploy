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
    ) -> tuple[list[Instrument], Optional[datetime], bool]:
        print(f"[NSEActiveResolver] Fetching instruments for {symbol} @ spot={spot_price}")
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

        # Nearest expiry >= reference_date
        target_expiry = next(
            (e for e in sorted(opts["expiry"].unique()) if e >= ref_dt),
            None,
        )
        if target_expiry is None:
            return [], None, True   # all expiries are past

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
    ) -> tuple[list[Instrument], Optional[datetime], bool]:
        underlying = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])
        print(f"[NSEExpiredResolver] symbol={symbol}, underlying={underlying}, spot={spot_price}")

        expiries = self._fetcher.fetch_expiries(underlying)
        if not expiries:
            return [], None, True

        # Find nearest expiry >= reference_date
        ref_dt          = datetime.strptime(reference_date, "%Y-%m-%d")
        target_expiry_s = None
        for exp in sorted(expiries):
            if datetime.strptime(str(exp), "%Y-%m-%d").date() >= ref_dt.date():
                target_expiry_s = str(exp)
                break
        if not target_expiry_s:
            target_expiry_s = str(sorted(expiries)[-1])
            print(f"[NSEExpiredResolver] No expiry >= {ref_dt.date()}, using latest: {target_expiry_s}")

        target_expiry_dt = datetime.strptime(target_expiry_s, "%Y-%m-%d")
        contracts        = self._fetcher.fetch_contracts(underlying, target_expiry_s)
        if not contracts:
            return [], target_expiry_dt, True

        cdf = pd.DataFrame([c if isinstance(c, dict) else c.__dict__ for c in contracts])
        cdf["strike"] = cdf["strike_price"].astype(float)
        cdf["type"]   = cdf["instrument_type"]

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
