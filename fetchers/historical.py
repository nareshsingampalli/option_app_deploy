"""HistoricalCandleFetcher — past candle data with date-range chunking."""

import signal
import time
from datetime import datetime, timedelta
import pytz

import pandas as pd
from upstox_client.rest import ApiException

from core.rate_limiter import rate_limited
from core.config import UPSTOX_RATE_LIMIT_CALLS, UPSTOX_RATE_LIMIT_PERIOD
from fetchers.base import BaseCandleFetcher


def _timeout_handler(signum, frame):
    raise TimeoutError("API request timed out")


class HistoricalCandleFetcher(BaseCandleFetcher):
    """Fetches historical candles; auto-chunks minute data into 30-day windows."""

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def fetch_single(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        to_date: str,
        from_date: str,
        timeout_secs: int = 90,
    ) -> pd.DataFrame | None:
        self._sync_token() # Sync with live MQ token
        from core.utils import retry_api_call
        
        @retry_api_call(max_retries=3)
        def _get():
            return self._history_api.get_historical_candle_data1(
                instrument_key, unit, interval, to_date, from_date
            )

        try:
            resp = _get()
            self._save_mock_response(resp, "historical", instrument_key)
            self.last_status = 200
            return self._process_response(resp)
        except ApiException as e:
            self.last_status = getattr(e, "status", None)
            print(f"[HistoricalFetcher] ApiException {instrument_key} (Status {self.last_status}): {e}")
            return None
        except Exception as e:
            print(f"[HistoricalFetcher] Error {instrument_key}: {e}")
            return None

    def fetch(
        self,
        instrument_key: str,
        timeframe: str = "days",
        interval: int = 1,
        lookback_days: int = 90,
    ) -> pd.DataFrame | None:
        unit = self._normalize_unit(timeframe)
        if unit in ("minutes", "hours") and lookback_days > 30:
            return self._fetch_chunked(instrument_key, unit, interval, lookback_days)
        to_date   = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return self.fetch_single(instrument_key, unit, interval, to_date, from_date)

    def _fetch_chunked(
        self, instrument_key: str, unit: str, interval: int, lookback_days: int, chunk=30
    ) -> pd.DataFrame | None:
        all_dfs, end_dt = [], datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days)
        cur_end  = end_dt
        while cur_end > start_dt:
            cur_start = max(cur_end - timedelta(days=chunk), start_dt)
            df = self.fetch_single(
                instrument_key, unit, interval,
                cur_end.strftime("%Y-%m-%d"), cur_start.strftime("%Y-%m-%d")
            )
            if df is not None and not df.empty:
                all_dfs.append(df)
            cur_end = cur_start - timedelta(days=1)
            time.sleep(0.5)
        if not all_dfs:
            return None
        combined = pd.concat(all_dfs)
        return combined[~combined.index.duplicated(keep="first")].sort_index()

    # ── Unified interface ────────────────────────────────────────────────────
    def _get_prev_trading_day(self, date_str: str) -> str:
        """Helper to find the date of the preceding trading session, skipping weekends & holidays."""
        from core.utils import get_last_trading_day
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Go back one day first, then find the last trading day from there
        prev_dt = dt - timedelta(days=1)
        return get_last_trading_day(prev_dt).strftime("%Y-%m-%d")

    def get_candles(self, instrument_key: str, date_str: str, expiry_dt=None) -> pd.DataFrame | None:
        self.used_fallback = False
        # Fetch from the previous trading day to provide a non-resetting baseline for ROC calculations.
        from_date = self._get_prev_trading_day(date_str)
        df = self.fetch_single(instrument_key, "minutes", self.interval, date_str, from_date)
        if (df is None or df.empty) and getattr(self, "last_status", None) != 429:
            df = self.fetch_single(instrument_key, "minutes", 1, date_str, from_date)
            if df is not None and not df.empty:
                self.used_fallback = True
        return df

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        # Same as get_candles: provide continuity across days.
        from_date = self._get_prev_trading_day(date_str)
        df = self.fetch_single(spot_key, "minutes", self.interval, date_str, from_date)
        if df is None or df.empty:
            df = self.fetch_single(spot_key, "minutes", 1, date_str, from_date)
        return df

    def get_spot_price_at(self, spot_key: str, date_str: str, time_str: str | None) -> float | None:
        """Return close price at target time, or daily close as fallback."""
        spot_price = None
        target_dt  = datetime.strptime(date_str, "%Y-%m-%d")

        if time_str:
            df = self.fetch_single(spot_key, "minutes", self.interval, date_str, date_str)
            if df is None or df.empty:
                df = self.fetch_single(spot_key, "minutes", 1, date_str, date_str)
                if df is not None and not df.empty:
                    self.used_fallback = True
            
            if df is not None and not df.empty:
                try:
                    # Target time (naive)
                    target_full = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                    
                    # Convert DF index to naive for comparison
                    df_naive = df.index.tz_localize(None) if hasattr(df.index, "tz_localize") else df.index
                    
                    idx = df_naive.get_indexer([target_full], method="nearest")[0]
                    spot_price = float(df.iloc[idx]["close"])
                    print(f"[HistoricalFetcher] Spot at {df.index[idx]}: {spot_price}")
                except Exception as e:
                    print(f"[HistoricalFetcher] Time lookup error: {e}")

        if spot_price is None:
            df_daily = self.fetch(spot_key, timeframe="days", lookback_days=15)
            if df_daily is not None and not df_daily.empty:
                dates = df_daily.index.strftime("%Y-%m-%d")
                if date_str in dates:
                    val = df_daily.loc[date_str]["close"]
                    spot_price = float(val.iloc[0] if hasattr(val, "iloc") else val)
                else:
                    td = target_dt if df_daily.index.tzinfo is None else target_dt.replace(tzinfo=df_daily.index.tzinfo)
                    past = df_daily[df_daily.index <= td]
                    if not past.empty:
                        spot_price = float(past["close"].iloc[-1])
        return spot_price
