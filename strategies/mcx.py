"""
Concrete MCX pipelines.

MCXLivePipeline       : intraday (IntradayCandleFetcher, MCXInstrumentResolver)
MCXHistoricalPipeline : historical (HistoricalCandleFetcher, MCXInstrumentResolver)

Spot key is resolved dynamically by MCXInstrumentResolver (nearest-expiry FUT).
MCX trading window: 09:00 – 23:30.
"""

from __future__ import annotations

from typing import Optional

from fetchers.base import BaseCandleFetcher
from resolvers.base import InstrumentResolver
from resolvers.mcx_resolver import MCXInstrumentResolver
from storage.base import StorageHandler
from core.config import MCX_MARKET_START, MCX_MARKET_END
from strategies.base import MarketDataPipeline


class _MCXBasePipeline(MarketDataPipeline):
    """Shared MCX base — resolves spot_key dynamically."""

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "CRUDEOIL"):
        super().__init__(fetcher, resolver, storage, MCX_MARKET_START, MCX_MARKET_END, "mcx")
        self._mcx_resolver = resolver   # MCXInstrumentResolver
        self._spot_key_cache: dict[str, str] = {}
        self.symbol = symbol.upper()

    def _get_spot_key(self, symbol: str, date_str: str) -> str:
        if symbol not in self._spot_key_cache:
            if isinstance(self._mcx_resolver, MCXInstrumentResolver):
                self._spot_key_cache[symbol] = self._mcx_resolver.get_spot_key(symbol, date_str)
            else:
                raise RuntimeError("MCX pipeline requires MCXInstrumentResolver")
        return self._spot_key_cache[symbol]


class MCXLivePipeline(_MCXBasePipeline):
    """Live intraday MCX data."""
    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "CRUDEOIL"):
        super().__init__(fetcher, resolver, storage, symbol=symbol)

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        # We don't know symbol here; pipeline.run() will call resolver first
        # Spot key is resolved lazily in build_spot_map; for live, return latest close
        return None   # resolved in run() override below

    def build_spot_map(self, target_date: str) -> dict:
        return {}

    def run(self, symbol: str, target_date: str, target_time: Optional[str] = None):
        """Override run to inject symbol into spot key lookup."""
        spot_key = self._get_spot_key(symbol, target_date)
        print(f"[MCXLive] spot_key={spot_key}")

        df = self.fetcher.get_spot_candles(spot_key, target_date)
        if df is None or df.empty:
            print("[MCXLive] No live spot data.")
            self._save([], None, target_date, target_time, None, False, symbol)
            return

        if target_time:
            from datetime import datetime
            try:
                ref_date = df.index[0].date().strftime("%Y-%m-%d")
                tgt_dt   = datetime.strptime(f"{ref_date} {target_time}", "%Y-%m-%d %H:%M")
                idx      = df.index.get_indexer([tgt_dt], method="nearest")[0]
                spot_p   = float(df.iloc[idx]["close"])
            except Exception as e:
                print(f"[MCXLive] Time lookup error: {e}")
                spot_p = float(df["close"].iloc[-1])
        else:
            spot_p = float(df["close"].iloc[-1])

        spot_map   = df["close"].to_dict()
        instruments, expiry_dt, is_expired = self.resolver.resolve(symbol, spot_p, target_date)
        if not instruments:
            self._save([], spot_p, target_date, target_time, expiry_dt, is_expired, symbol)
            return

        rows, used_fallback = self._process_all(instruments, spot_map, target_date)
        self._save(rows, spot_p, target_date, target_time, expiry_dt, is_expired, symbol, used_fallback)


class MCXHistoricalPipeline(_MCXBasePipeline):
    """Historical MCX data."""
    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "CRUDEOIL"):
        super().__init__(fetcher, resolver, storage, symbol=symbol)

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        return None   # resolved in run() override

    def build_spot_map(self, target_date: str) -> dict:
        return {}

    def run(self, symbol: str, target_date: str, target_time: Optional[str] = None):
        spot_key = self._get_spot_key(symbol, target_date)
        from fetchers.historical import HistoricalCandleFetcher
        hist = self.fetcher if isinstance(self.fetcher, HistoricalCandleFetcher) else HistoricalCandleFetcher()

        spot_p = hist.get_spot_price_at(spot_key, target_date, target_time)
        if not spot_p:
            print("[MCXHistorical] No spot price.")
            self._save([], None, target_date, target_time, None, False, symbol)
            return

        spot_map   = {}
        df_map = hist.get_spot_candles(spot_key, target_date)
        if df_map is not None and not df_map.empty:
            spot_map = df_map["close"].to_dict()

        instruments, expiry_dt, is_expired = self.resolver.resolve(symbol, spot_p, target_date)
        if not instruments:
            self._save([], spot_p, target_date, target_time, expiry_dt, is_expired, symbol)
            return

        rows, used_fallback = self._process_all(instruments, spot_map, target_date)
        self._save(rows, spot_p, target_date, target_time, expiry_dt, is_expired, symbol, used_fallback)
