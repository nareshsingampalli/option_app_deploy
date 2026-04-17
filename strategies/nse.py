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

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "NIFTY", expiry_offset: int = 0):
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option", expiry_offset=expiry_offset)
        self._spot_key = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        print(f"[NSELive] Fetching spot for {target_date} {target_time or '(Latest)'}...")
        df = self.fetcher.get_spot_candles(self._spot_key, target_date)
        if df is None or df.empty:
            return None
        
        if target_time:
            # Find row exactly matching HH:MM (ignoring seconds)
            match = df[df.index.astype(str).str.contains(target_time)]
            if not match.empty:
                return float(match["close"].iloc[-1])
        
        # Default to latest
        return float(df["close"].iloc[-1])

    def build_spot_map(self, target_date: str) -> dict:
        df = self.fetcher.get_spot_candles(self._spot_key, target_date)
        return df["close"].to_dict() if df is not None and not df.empty else {}


class NSEHistoricalPipeline(MarketDataPipeline):
    """Historical NSE data (active contracts)."""

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "NIFTY", expiry_offset: int = 0):
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option", expiry_offset=expiry_offset)
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

    def __init__(self, fetcher: BaseCandleFetcher, resolver: InstrumentResolver, storage: StorageHandler, symbol: str = "NIFTY", expiry_offset: int = 0):
        super().__init__(fetcher, resolver, storage, NSE_MARKET_START, NSE_MARKET_END, "option", expiry_offset=expiry_offset)
        self._spot_key   = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])
        self._hist_fetch = HistoricalCandleFetcher()   # always needed for spot

    def fetch_spot_price(self, target_date: str, target_time: Optional[str]) -> Optional[float]:
        return self._hist_fetch.get_spot_price_at(self._spot_key, target_date, target_time)

    def build_spot_map(self, target_date: str) -> dict:
        df = self._hist_fetch.get_spot_candles(self._spot_key, target_date)
        return df["close"].to_dict() if df is not None and not df.empty else {}

    def _process_all(self, instruments: list[Instrument], spot_map: dict, filter_date: str) -> tuple[list[dict], bool]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        rows: list[dict] = []
        any_fallback = False

        def process_one(inst: Instrument):
            fetcher_to_use = self.fetcher
            if inst.key.count("|") < 2:
                 fetcher_to_use = self._hist_fetch
            
            print(f"[Pipeline] Processing {inst.symbol} using {type(fetcher_to_use).__name__}...")
            try:
                df = fetcher_to_use.get_candles(inst.key, filter_date, expiry_dt=inst.expiry)
                
                # Hybrid Fallback: If expired fetcher fails, try historical (active series)
                if (df is None or df.empty) and fetcher_to_use != self._hist_fetch:
                    print(f"[Pipeline] {inst.symbol} not in Expired API. Falling back to Historical API...")
                    df = self._hist_fetch.get_candles(inst.key, filter_date, expiry_dt=inst.expiry)

                if df is None or df.empty:
                    return [], False
                
                fallback = getattr(fetcher_to_use, "used_fallback", False)
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

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_inst = {executor.submit(process_one, inst): inst for inst in instruments}
            for future in as_completed(future_to_inst):
                result_rows, fallback = future.result()
                if fallback:
                    any_fallback = True
                rows.extend(result_rows)

        return rows, any_fallback
