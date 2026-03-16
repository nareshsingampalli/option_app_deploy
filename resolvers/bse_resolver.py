"""
BSEResolver
-----------
Resolves active BSE index option contracts (e.g. SENSEX, BANKEX).
Similar to NSE resolver but pointed to BSE instrument list.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from core.config import BSE_INSTRUMENT_URL
from resolvers.base import Instrument, InstrumentResolver

class BSEActiveResolver(InstrumentResolver):
    """Resolves active BSE option contracts from the instruments CSV/JSON."""

    def resolve(
        self,
        symbol:         str,
        spot_price:     float,
        reference_date: str,
        num_strikes:    int = 3,
    ) -> tuple[list[Instrument], Optional[datetime], bool]:
        print(f"[BSEActiveResolver] Fetching instruments for {symbol} @ spot={spot_price}")
        from core.utils import get_instrument_df
        
        # BSE instrument list usually comes as JSON.gz
        df = get_instrument_df(BSE_INSTRUMENT_URL, "BSE")

        # BSE instruments: instrument_type is 'CE' or 'PE' directly (not 'OPTIDX')
        sym_upper = symbol.upper()
        opts = df[
            (df["instrument_type"].isin(["CE", "PE"])) &
            (df["name"] == sym_upper)
        ].copy()
        
        # Rename columns if necessary
        if "tradingsymbol" in opts.columns:
            opts = opts.rename(columns={"tradingsymbol": "trading_symbol"})
        if "strike" in opts.columns:
            opts = opts.rename(columns={"strike": "strike_price"})

        if opts.empty:
            print(f"[BSEActiveResolver] No CE/PE instruments found for {sym_upper}")
            return [], None, False

        # BSE Expiry is often in epoch milliseconds
        if pd.api.types.is_numeric_dtype(opts["expiry"]):
            opts["expiry"] = pd.to_datetime(opts["expiry"], unit="ms")
        else:
            opts["expiry"] = pd.to_datetime(opts["expiry"])
            
        ref_dt = pd.to_datetime(reference_date)

        # Nearest expiry >= reference_date
        expiries = sorted(opts["expiry"].unique())
        target_expiry = next((e for e in expiries if e >= ref_dt), None)
        
        if target_expiry is None:
            return [], None, True # all expiries are past

        expiry_df = opts[opts["expiry"] == target_expiry].copy()
        strikes   = sorted(expiry_df["strike_price"].unique())
        
        if not strikes:
            return [], target_expiry.to_pydatetime(), False
            
        idx       = int(np.abs(np.array(strikes) - spot_price).argmin())
        selected  = strikes[max(0, idx - num_strikes): idx + num_strikes + 1]

        instruments: list[Instrument] = []
        for strike in selected:
            for opt_type in ("CE", "PE"):
                rows = expiry_df[
                    (expiry_df["strike_price"] == strike) &
                    (expiry_df["instrument_type"] == opt_type)
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

        print(f"[BSEActiveResolver] {len(instruments)} instruments selected")
        return instruments, target_expiry.to_pydatetime(), False
