"""
BSE Strategies
--------------
Live and Historical pipelines for BSE indices (SENSEX, BANKEX).
"""

from __future__ import annotations
from typing import Optional

from fetchers.historical import HistoricalCandleFetcher
from fetchers.base import BaseCandleFetcher
from resolvers.base import InstrumentResolver
from storage.base import StorageHandler
from core.config import BSE_INDEX_KEYS, NSE_MARKET_START, NSE_MARKET_END
from strategies.base import MarketDataPipeline


class BSELivePipeline(MarketDataPipeline):
    """Live intraday BSE data."""

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "SENSEX", expiry_offset: int = 0):
        # BSE uses same trading hours as NSE (mostly)
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option", expiry_offset=expiry_offset)
        self._spot_key = BSE_INDEX_KEYS.get(symbol.upper(), BSE_INDEX_KEYS["SENSEX"])

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        print("[BSELive] Fetching live spot...")
        df = self.fetcher.get_spot_candles(self._spot_key, target_date)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    def build_spot_map(self, target_date: str) -> dict:
        df = self.fetcher.get_spot_candles(self._spot_key, target_date)
        return df["close"].to_dict() if df is not None and not df.empty else {}


class BSEHistoricalPipeline(MarketDataPipeline):
    """Historical BSE data."""

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "SENSEX", expiry_offset: int = 0):
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option", expiry_offset=expiry_offset)
        self._spot_key = BSE_INDEX_KEYS.get(symbol.upper(), BSE_INDEX_KEYS["SENSEX"])
        assert isinstance(fetcher, HistoricalCandleFetcher), "BSEHistoricalPipeline requires a HistoricalCandleFetcher"
        self._hist: HistoricalCandleFetcher = fetcher

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        return self._hist.get_spot_price_at(self._spot_key, target_date, target_time)

    def build_spot_map(self, target_date: str) -> dict:
        df = self._hist.get_spot_candles(self._spot_key, target_date)
        return df["close"].to_dict() if df is not None and not df.empty else {}
