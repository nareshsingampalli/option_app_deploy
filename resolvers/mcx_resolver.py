"""
MCXInstrumentResolver
---------------------
Dynamic resolver — NO hardcoded MCX_FO keys.

Flow
----
1. Download MCX.json.gz from Upstox assets.
2. Find FUT instruments for the commodity → sort_values(expiry ASC) → take first = spot_key.
3. Apply D-1 rule: if today is within 1 day of expiry, use next expiry.
4. Use precise_pattern to filter CE/PE instruments for that expiry.
5. Select strikes around ATM (step = MCX_STRIKE_STEP = 50).
"""

from __future__ import annotations

import gzip
import io
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

from core.config import MCX_INSTRUMENT_URL, MCX_STRIKE_STEP, DEFAULT_NUM_STRIKES
from core.exceptions import InstrumentResolutionError
from resolvers.base import Instrument, InstrumentResolver


class MCXInstrumentResolver(InstrumentResolver):
    """Resolves MCX option instruments dynamically from MCX.json.gz."""

    # Resolved spot key cache: {commodity: (instrument_key, expiry_dt)}
    _spot_cache: dict[str, tuple[str, pd.Timestamp]] = {}

    def resolve(
        self,
        symbol:         str,
        spot_price:     float,
        reference_date: str,
        num_strikes:    int = DEFAULT_NUM_STRIKES,
    ) -> tuple[list[Instrument], Optional[datetime], bool]:
        commodity = symbol.upper()
        print(f"[MCXResolver] commodity={commodity}, spot={spot_price}, date={reference_date}")

        df_all = self._download_instruments()

        # ── Step 1: Find spot key (nearest-expiry FUT) ───────────────────────
        spot_key, target_expiry = self._resolve_spot_key(commodity, df_all, reference_date)
        print(f"[MCXResolver] spot_key={spot_key}, expiry={target_expiry.date()}")

        # ── Step 2: Compute ATM strikes ──────────────────────────────────────
        atm = round(spot_price / MCX_STRIKE_STEP) * MCX_STRIKE_STEP
        target_strikes = [atm + MCX_STRIKE_STEP * i for i in range(-num_strikes, num_strikes + 1)]
        print(f"[MCXResolver] ATM={atm}, strikes={target_strikes}")

        # ── Step 3: Filter options with precise_pattern ──────────────────────
        instruments = self._resolve_options(commodity, target_expiry, target_strikes, df_all)

        if not instruments:
            return [], target_expiry.to_pydatetime(), False

        # Use the actual option expiry for the series status
        actual_expiry = instruments[0].expiry
        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
        
        # Extract date for comparison
        is_expired = ref_dt.date() > actual_expiry.date()

        return instruments, actual_expiry, is_expired

    # ── Helpers ──────────────────────────────────────────────────────────────
    def get_spot_key(self, commodity: str, reference_date: str) -> str:
        """Public helper: return just the spot instrument key for a commodity."""
        df_all = self._download_instruments()
        key, _ = self._resolve_spot_key(commodity, df_all, reference_date)
        return key

    def _download_instruments(self) -> pd.DataFrame:
        from core.utils import get_instrument_df
        return get_instrument_df(MCX_INSTRUMENT_URL, "MCX")

    def _resolve_spot_key(
        self, commodity: str, df_all: pd.DataFrame, reference_date: str
    ) -> tuple[str, pd.Timestamp]:
        """
        Find nearest-expiry FUT instrument for the commodity.
        Extracts date from trading_symbol (e.g. 26 MAR 26) instead of using expiry col.
        Rolls to next month if reference_date >= (symbol_date - 1 day).
        """
        ref_dt = pd.Timestamp(reference_date).normalize()
        # Start search from the month of the reference date
        search_dt = ref_dt
        
        while True:
            month_str = search_dt.strftime("%b").upper()
            year_str  = search_dt.strftime("%y")
            # Pattern to find the future and extract the day
            pattern = rf"^{commodity + ' '}FUT (\d{{1,2}}) {month_str} {year_str}$"
            
            # Find candidate in df_all
            matches = df_all[df_all["trading_symbol"].str.match(pattern)].copy()
            
            if matches.empty:
                # No future for this month? Jump to next month and try again.
                search_dt = (search_dt.replace(day=1) + pd.DateOffset(months=1))
                if (search_dt - ref_dt).days > 365:
                    raise InstrumentResolutionError(f"No FUT found for {commodity} in the next 12 months.")
                continue
            
            # Found the future for this month. Pick the first one string-wise if multiple.
            match = matches.iloc[0]
            sym = match["trading_symbol"]
            
            # Extract day from symbol
            match_obj = re.match(pattern, sym)
            day_str = match_obj.group(1)
            
            # Construct date from symbol string: e.g. "26 MAR 26"
            expiry_from_sym = pd.to_datetime(f"{day_str} {month_str} {year_str}", format="%d %b %y").normalize()
            
            # ROLL RULE: All MCX instruments expire on exp - 1
            # If today is exp-1 or later, we roll to next month.
            roll_cutoff = expiry_from_sym - pd.Timedelta(days=1)
            
            if ref_dt < roll_cutoff:
                # We haven't hit the rollover yet.
                return match["instrument_key"], expiry_from_sym
            else:
                # It is the day before (or after) the string expiry. Roll to next month.
                search_dt = (search_dt.replace(day=1) + pd.DateOffset(months=1))

    def _resolve_options(
        self,
        commodity:      str,
        target_expiry:  pd.Timestamp,
        target_strikes: list[float],
        df_all:         pd.DataFrame,
    ) -> list[Instrument]:
        month = target_expiry.strftime("%b").upper()
        year  = target_expiry.strftime("%y")

        # Matches: "CRUDEOIL 6600 CE 17 MAR 26" as requested
        precise_pattern = rf"^{commodity} \d+ (?:CE|PE) \d{{1,2}} {month} {year}$"

        opts = df_all[
            df_all["instrument_type"].isin(["CE", "PE"]) &
            df_all["trading_symbol"].str.contains(precise_pattern, regex=True)
        ].copy()

        # Extract expiry date from symbol string as requested: "NATURALGAS 390 PE 24 MAR 26"
        def extract_expiry_from_sym(sym_str):
            m = re.search(rf"(\d{{1,2}}) {month} {year}$", sym_str)
            if m:
                day = m.group(1)
                return pd.to_datetime(f"{day} {month} {year}", format="%d %b %y").normalize()
            return target_expiry # Fallback

        opts["expiry_dt"] = opts["trading_symbol"].apply(extract_expiry_from_sym)
        opts["strike"] = opts["strike_price"].astype(float)

        if opts.empty:
            print(f"[MCXResolver] No options matched pattern for {month} {year}")
            return []

        instruments: list[Instrument] = []
        for strike in target_strikes:
            for opt_type in ("CE", "PE"):
                match = opts[
                    (opts["instrument_type"] == opt_type) &
                    (opts["strike"] == float(strike))
                ]
                if match.empty:
                    # Nearest available strike fallback
                    subset = opts[opts["instrument_type"] == opt_type].copy()
                    subset["dist"] = (subset["strike"] - strike).abs()
                    match = subset.nsmallest(1, "dist")
                    if match.empty:
                        continue
                    found = match.iloc[0]["strike"]
                    print(f"[MCXResolver] Strike {strike}{opt_type} not found, using nearest: {found}")

                row = match.iloc[0]
                sym = row["trading_symbol"]
                m   = re.search(r"(?:CE|PE)\s+(.+)$", sym)
                expiry_str = m.group(1).strip() if m else target_expiry.strftime("%d %b %y").upper()

                instruments.append(Instrument(
                    key=row["instrument_key"],
                    symbol=sym,
                    strike=float(row["strike"]),
                    option_type=opt_type,
                    expiry=row["expiry_dt"].to_pydatetime(),
                    expiry_str=expiry_str,
                ))

        print(f"[MCXResolver] {len(instruments)} instruments selected")
        return instruments
