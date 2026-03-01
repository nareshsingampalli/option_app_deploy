"""
MarketDataPipeline — Template Method Pattern
--------------------------------------------
Defines the skeleton of the option chain data pipeline.
Concrete subclasses override only the steps that differ.

Pipeline skeleton (run)
────────────────────────
  1. fetch_spot_price()     ← abstract: each mode fetches differently
  2. build_spot_map()       ← abstract: intraday vs historical 5-min
  3. resolver.resolve()     ← injected InstrumentResolver
  4. _process_all()         ← concrete: shared IV/ROC logic
  5. storage_chain.save()   ← injected StorageBackend chain

Dependency injection
--------------------
  • fetcher   → BaseCandleFetcher (from Factory)
  • resolver  → InstrumentResolver (NSEActive/NSEExpired/MCX)
  • storage   → StorageHandler chain (File → DB)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.black_scholes import implied_volatility
from core.config import RISK_FREE_RATE, NSE_MARKET_START, NSE_MARKET_END
from fetchers.base import BaseCandleFetcher
from resolvers.base import Instrument, InstrumentResolver
from storage.base import SaveContext, StorageHandler


class MarketDataPipeline(ABC):
    """
    Template Method base class.

    Parameters (all injected)
    -------------------------
    fetcher      : BaseCandleFetcher — how to fetch option candles
    resolver     : InstrumentResolver — how to find instruments
    storage      : StorageHandler     — where to save results
    market_start : trading start time string 'HH:MM'
    market_end   : trading end time string 'HH:MM'
    prefix       : file prefix ('option' | 'mcx')
    """

    def __init__(
        self,
        fetcher:      BaseCandleFetcher,
        resolver:     InstrumentResolver,
        storage:      StorageHandler,
        market_start: str = NSE_MARKET_START,
        market_end:   str = NSE_MARKET_END,
        prefix:       str = "option",
    ):
        self.fetcher      = fetcher
        self.resolver     = resolver
        self.storage      = storage
        self.market_start = datetime.strptime(market_start, "%H:%M").time()
        self.market_end   = datetime.strptime(market_end,   "%H:%M").time()
        self.prefix       = prefix
        self.symbol       = ""  # Will be set in run() if needed, but run() already takes it

    # ── Template method (orchestrator) ───────────────────────────────────────
    def run(self, symbol: str, target_date: str, target_time: Optional[str] = None):
        print(f"\n[Pipeline] Starting: {symbol} | {target_date} {target_time or '(EOD)'}")

        # Step 1 — Spot price
        spot_price = self.fetch_spot_price(target_date, target_time)
        if not spot_price:
            print("[Pipeline] Could not resolve spot price. Aborting.")
            self._save([], spot_price, target_date, target_time, None, False, symbol)
            return

        # Step 2 — Resolve instruments
        instruments, expiry_dt, is_expired = self.resolver.resolve(
            symbol, spot_price, target_date
        )
        if not instruments:
            print("[Pipeline] No instruments found.")
            self._save([], spot_price, target_date, target_time, expiry_dt, is_expired, symbol)
            return

        # Step 3 — Build spot map for IV
        spot_map = self.build_spot_map(target_date)

        # Step 4 — Process each instrument
        rows, used_fallback = self._process_all(instruments, spot_map, target_date)

        # Step 5 — Save
        self._save(rows, spot_price, target_date, target_time, expiry_dt, is_expired, symbol, used_fallback)

    # ── Abstract hooks ────────────────────────────────────────────────────────
    @abstractmethod
    def fetch_spot_price(
        self, target_date: str, target_time: Optional[str]
    ) -> Optional[float]:
        """Return the spot/underlying price used for ATM selection."""

    @abstractmethod
    def build_spot_map(self, target_date: str) -> dict:
        """Return {pd.Timestamp: float} mapping used for IV calculation."""

    # ── Concrete shared steps ─────────────────────────────────────────────────
    def _process_all(
        self, instruments: list[Instrument], spot_map: dict, filter_date: str
    ) -> tuple[list[dict], bool]:
        rows: list[dict] = []
        any_fallback = getattr(self.fetcher, "used_fallback", False)

        for inst in instruments:
            print(f"[Pipeline] Processing {inst.symbol}...")
            try:
                df = self.fetcher.get_candles(inst.key, filter_date, expiry_dt=inst.expiry)
                if df is None or df.empty:
                    continue
                if getattr(self.fetcher, "used_fallback", False):
                    any_fallback = True
                processed = self._process_instrument(df, spot_map, inst, filter_date)
                for row in processed:
                    row["symbol"]      = inst.symbol
                    row["strike"]      = inst.strike
                    row["option_type"] = inst.option_type
                    row["expiry_text"] = inst.expiry_str
                    rows.append(row)
            except Exception as e:
                print(f"[Pipeline] Error on {inst.symbol}: {e}")
        return rows, any_fallback

    def _process_instrument(
        self,
        df:          pd.DataFrame,
        spot_map:    dict,
        inst:        Instrument,
        filter_date: Optional[str] = None,
    ) -> list[dict]:
        """Compute IV, ROC, COI and return records."""
        df = df.rename(columns={"close": "ltp", "open_interest": "oi"})
        df["change_in_oi"] = df["oi"].diff().fillna(0)

        expiry_with_time = inst.expiry.replace(hour=15, minute=30)
        iv_list: list[float] = []

        for idx, row in df.iterrows():
            spot_p = spot_map.get(idx)
            if spot_p:
                exp_t = expiry_with_time
                if idx.tzinfo and not exp_t.tzinfo:
                    exp_t = exp_t.replace(tzinfo=idx.tzinfo)
                T = (exp_t - idx).total_seconds() / (365 * 24 * 3600)
                iv = implied_volatility(row["ltp"], spot_p, inst.strike, T, RISK_FREE_RATE, inst.option_type) if T > 0 else 0.0
                iv_list.append(round(iv * 100, 2))
            else:
                iv_list.append(0.0)

        df["iv"]             = iv_list
        df["change_in_ltp"]  = df["ltp"].diff().fillna(0)
        df["roc_oi"]         = (df["oi"].pct_change()     * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        df["roc_volume"]     = (df["volume"].pct_change() * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        df["roc_iv"]         = (df["iv"].pct_change()     * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        df["coi_vol_ratio"]  = (df["change_in_oi"] / df["volume"]).replace([np.inf, -np.inf], 0).fillna(0).round(4)
        df["spot_price"]     = df.index.map(spot_map)
        df["spot_price"]     = df["spot_price"].replace(0, pd.NA).ffill().bfill().fillna(0)

        result = df[["ltp", "change_in_ltp", "roc_oi", "roc_volume", "roc_iv", "coi_vol_ratio", "spot_price"]].reset_index()

        # Time-window filter
        result = result[
            (result["date"].dt.time >= self.market_start) &
            (result["date"].dt.time <= self.market_end)
        ]
        if filter_date:
            result = result[result["date"].dt.strftime("%Y-%m-%d") == filter_date]

        result["date"] = result["date"].astype(str)
        return result.to_dict(orient="records")

    def _save(self, rows, spot_price, date_str, time_str, expiry_dt, is_expired, symbol, is_fallback=False):
        ctx = SaveContext(
            rows=rows,
            spot_price=spot_price,
            date_str=date_str,
            time_str=time_str,
            expiry_dt=expiry_dt,
            is_expired=is_expired,
            prefix=self.prefix,
            symbol=symbol,
            is_fallback=is_fallback,
        )
        self.storage.handle(ctx)
