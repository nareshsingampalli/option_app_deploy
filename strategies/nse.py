"""
Concrete NSE pipelines — override only the spot-fetching hooks.

NSELivePipeline      : intraday (IntradayCandleFetcher, NSEActiveResolver)
NSEHistoricalPipeline: historical (HistoricalCandleFetcher, NSEActiveResolver)
NSEExpiredPipeline   : expired   (ExpiredCandleFetcher,   NSEExpiredResolver)
                       + keeps a HistoricalCandleFetcher for spot index data
"""

from __future__ import annotations

from typing import Optional

from fetchers.historical import HistoricalCandleFetcher
from fetchers.base import BaseCandleFetcher
from resolvers.base import InstrumentResolver
from storage.base import StorageHandler
from core.config import NSE_INDEX_KEYS, NSE_MARKET_START, NSE_MARKET_END
from strategies.base import MarketDataPipeline


class NSELivePipeline(MarketDataPipeline):
    """Live intraday NSE data."""

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "NIFTY"):
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option")
        self._spot_key = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        print("[NSELive] Fetching live spot...")
        df = self.fetcher.get_spot_candles(self._spot_key, target_date)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    def build_spot_map(self, target_date: str) -> dict:
        df = self.fetcher.get_spot_candles(self._spot_key, target_date)
        return df["close"].to_dict() if df is not None and not df.empty else {}


class NSEHistoricalPipeline(MarketDataPipeline):
    """Historical NSE data (active contracts)."""

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "NIFTY"):
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option")
        self._spot_key = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])
        # We know the factory always returns a HistoricalCandleFetcher for this pipeline
        assert isinstance(fetcher, HistoricalCandleFetcher), \
            "NSEHistoricalPipeline requires a HistoricalCandleFetcher"
        self._hist: HistoricalCandleFetcher = fetcher

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        return self._hist.get_spot_price_at(self._spot_key, target_date, target_time)

    def build_spot_map(self, target_date: str) -> dict:
        df = self._hist.get_spot_candles(self._spot_key, target_date)
        return df["close"].to_dict() if df is not None and not df.empty else {}


class NSEExpiredPipeline(MarketDataPipeline):
    """
    Expired NSE contract data.
    Uses ExpiredCandleFetcher for option candles,
    HistoricalCandleFetcher for spot index (not available via expired API).
    """

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "NIFTY"):
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option")
        self._spot_key   = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])
        self._hist_fetch = HistoricalCandleFetcher()   # always needed for spot

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        return self._hist_fetch.get_spot_price_at(self._spot_key, target_date, target_time)

    def build_spot_map(self, target_date: str) -> dict:
        df = self._hist_fetch.get_spot_candles(self._spot_key, target_date)
        return df["close"].to_dict() if df is not None and not df.empty else {}
