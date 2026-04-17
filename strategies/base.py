"""
MarketDataPipeline — Template Method Pattern
--------------------------------------------
Defines the skeleton of the option chain data pipeline.
Concrete subclasses override only the steps that differ.

Pipeline skeleton (run)
-----------------------
  1. fetch_spot_price()     <- abstract: each mode fetches differently
  2. build_spot_map()       <- abstract: intraday vs historical 5-min
  3. resolver.resolve()     <- injected InstrumentResolver
  4. _process_all()         <- concrete: shared IV/ROC logic
  5. storage_chain.save()   <- injected StorageBackend chain

Dependency injection
--------------------
  * fetcher   -> BaseCandleFetcher (from Factory)
  * resolver  -> InstrumentResolver (NSEActive/NSEExpired/MCX)
  * storage   -> StorageHandler chain (File -> DB)
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
from strategies.normalization import NormalizationFactory


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
        expiry_offset: int = 0,
    ):
        self.fetcher      = fetcher
        self.resolver     = resolver
        self.storage      = storage
        self.market_start = datetime.strptime(market_start, "%H:%M").time()
        self.market_end   = datetime.strptime(market_end,   "%H:%M").time()
        self.prefix       = prefix
        self.expiry_offset = expiry_offset
        self.symbol       = ""  # Will be set in run() if needed
        self._inst_observers = []
        
        # Self-subscribe the fetcher's warmup if available
        self.subscribe_instruments(self.fetcher.warmup_cache)

    def subscribe_instruments(self, callback):
        """Register a subscriber (Observer) for instrument list changes."""
        self._inst_observers.append(callback)

    def _notify_instruments(self, instruments, target_date):
        """Notify all observers (subscribers) that the instrument list has been updated."""
        for observer in self._inst_observers:
            try:
                observer(instruments, target_date)
            except Exception as e:
                print(f"[Pipeline] Observer error: {e}")

    # ── Template method (orchestrator) ───────────────────────────────────────
    def run(self, symbol: str, target_date: str, target_time: Optional[str] = None):
        import time
        start_time = time.time()
        print(f"\n[Pipeline] Starting: {symbol} | {target_date} {target_time or '(EOD)'}")

        # Step 1 — Spot price
        t0 = time.time()
        spot_price = self.fetch_spot_price(target_date, target_time)
        t_spot = time.time() - t0
        
        if not spot_price:
            print(f"[Pipeline] Could not resolve spot price ({t_spot:.2f}s). Aborting.")
            self._save([], spot_price, target_date, target_time, None, False, symbol)
            return

        # Step 2 — Resolve instruments
        t0 = time.time()
        instruments, expiry_dt, is_expired = self.resolver.resolve(
            symbol, spot_price, target_date, expiry_offset=self.expiry_offset
        )
        t_resolve = time.time() - t0
        
        if not instruments:
            print(f"[Pipeline] No instruments found ({t_resolve:.2f}s).")
            self._save([], spot_price, target_date, target_time, expiry_dt, is_expired, symbol)
            return

        # Step 3 — Notify subscribers (e.g. parallel fetch baselines for Intraday)
        self._notify_instruments(instruments, target_date)

        # Step 4 — Build spot map for IV
        t0 = time.time()
        spot_map = self.build_spot_map(target_date)
        t_spot_map = time.time() - t0

        # Step 5 — Process each instrument (Now Parallel)
        t0 = time.time()
        rows, used_fallback = self._process_all(instruments, spot_map, target_date)
        t_process = time.time() - t0

        # Step 6 — Save
        t0 = time.time()
        self._save(rows, spot_price, target_date, target_time, expiry_dt, is_expired, symbol, used_fallback)
        t_save = time.time() - t0

        total = time.time() - start_time
        print(f"[Pipeline] Finished {symbol} in {total:.2f}s | Spot:{t_spot:.2f}s | Resolve:{t_resolve:.2f}s | Map:{t_spot_map:.2f}s | ParallelProc:{t_process:.2f}s | Save:{t_save:.2f}s")

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
        import eventlet
        from eventlet import GreenPool
        
        rows: list[dict] = []
        any_fallback = getattr(self.fetcher, "used_fallback", False)

        def process_one(inst: Instrument):
            print(f"[Pipeline] Processing {inst.symbol}...")
            # Yield to the hub before starting a new heavy task
            eventlet.sleep(0)
            try:
                df = self.fetcher.get_candles(inst.key, filter_date, expiry_dt=inst.expiry)
                if df is None or df.empty:
                    return [], False
                
                fallback = getattr(self.fetcher, "used_fallback", False)
                processed = self._process_instrument(df, spot_map, inst, filter_date)
                
                for row in processed:
                    row["symbol"]      = inst.symbol
                    row["strike"]      = inst.strike
                    row["option_type"] = inst.option_type
                    row["expiry_text"] = inst.expiry_str
                
                return processed, fallback
            except Exception as e:
                print(f"[Pipeline] Error on {inst.symbol}: {e}")
                return [], False

        # Use GreenPool for cooperative multitasking (yields during I/O)
        pool = GreenPool(size=10)
        for result_rows, fallback in pool.imap(process_one, instruments):
            if fallback:
                any_fallback = True
            rows.extend(result_rows)
            # Yield control back to hub after every instrument processed
            eventlet.sleep(0)

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

        expiry_with_time = inst.expiry.replace(hour=self.market_end.hour, minute=self.market_end.minute)
        iv_list: list[float] = []

        import eventlet
        for idx, row in df.iterrows():
            # Yield every candle iteration to prevent starving the hub
            eventlet.sleep(0)
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
        
        # ── ROC Calculations with cleaning & smoothing ──────────────────────────
        def process_roc(series, strategy='default', **kwargs):
            """
            Open-Closed Principle: This method now picks a dynamic strategy
            for normalization without changing its internal signature.
            """
            return NormalizationFactory.get_strategy(strategy).process(series, **kwargs)

        # Optimal strategy for ROC IV: 'soft_clip' squashes spikes while keeping neg/pos balance
        df["roc_iv"]         = process_roc(df["iv"], strategy='soft_clip', mask_series=df["iv"], mask_threshold=0.05)
        
        # Compute OI/Volume ROC on full data (including baseline)
        # Percentage changes now pick up the jump from yesterday's baseline if available.
        df["roc_oi"]         = process_roc(df["oi"], strategy='soft_clip')
        df["roc_volume"]     = process_roc(df["volume"], strategy='soft_clip')
        
        df["coi_vol_ratio"]  = (df["change_in_oi"] / df["volume"]).replace([np.inf, -np.inf], 0).fillna(0).round(4)
        df["spot_price"]     = df.index.map(spot_map)
        df["spot_price"]     = df["spot_price"].replace(0, pd.NA).ffill().bfill().fillna(0)

        # Include necessary metrics for dashboard
        result = df[["ltp", "change_in_ltp", "oi", "volume", "roc_iv", "roc_oi", "roc_volume", "coi_vol_ratio", "spot_price"]].reset_index()

        # ── Time-window filtering & user requested Warmup (2 candles) ─────────
        # Filter for the specific date first
        if filter_date:
            result = result[result["date"].dt.strftime("%Y-%m-%d") == filter_date]

        # Extract only the candles within market hours
        # Note: self.market_start is used here, which is now 09:14 for NSE to allow
        # the 09:15 candle to compute a non-zero change from a preceding value if present.
        session_data = result[
            (result["date"].dt.time >= self.market_start) &
            (result["date"].dt.time <= self.market_end)
        ].copy()

        # User request: "don't skip"
        # Return the full session data including the first available candles.
        result = session_data.copy()

        # Fill NaNs from ROC calculations
        for col in ["roc_oi", "roc_volume", "roc_iv"]:
             if col in result.columns:
                  result[col] = result[col].fillna(0).round(4)

        result.drop(columns=["oi", "volume"], inplace=True)

        # Shift timestamp to candle CLOSE time so chart shows
        # "close price at close time" rather than at the open time.
        result["date"] = result["date"] + pd.Timedelta(minutes=self.fetcher.interval)

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
            interval=self.fetcher.interval,
            is_fallback=is_fallback,
            next_expiry=self.expiry_offset > 0,
        )
        self.storage.handle(ctx)
