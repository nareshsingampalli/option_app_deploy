"""IntradayCandleFetcher — today's live candle data."""

import pandas as pd
from upstox_client.rest import ApiException

from core.rate_limiter import rate_limited
from core.config import UPSTOX_RATE_LIMIT_CALLS, UPSTOX_RATE_LIMIT_PERIOD
from fetchers.base import BaseCandleFetcher


class IntradayCandleFetcher(BaseCandleFetcher):
    """Fetches real-time 5-min candles for the current trading day."""

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def _fetch(self, instrument_key: str, unit: str, interval: int) -> pd.DataFrame | None:
        from core.utils import retry_api_call
        
        @retry_api_call(max_retries=3)
        def _get():
            return self._history_api.get_intra_day_candle_data(
                instrument_key, unit, interval
            )

        try:
            resp = _get()
            self._save_mock_response(resp, "intraday", instrument_key)
            self.last_status = 200
            return self._process_response(resp)
        except ApiException as e:
            self.last_status = getattr(e, "status", None)
            print(f"[IntradayFetcher] ApiException {instrument_key} (Status {self.last_status}): {e}")
            return None

    def get_candles(self, instrument_key: str, date_str: str, expiry_dt=None) -> pd.DataFrame | None:
        df = self._fetch(instrument_key, "minutes", self.interval)
        if df is None or df.empty:
            print(f"[IntradayFetcher] {self.interval}-min unavailable, falling back to 1-min")
            df = self._fetch(instrument_key, "minutes", 1)
        
        if df is not None and not df.empty:
            try:
                # Fetch the last candle of the previous trading day to provide a non-resetting baseline for ROC.
                from fetchers.historical import HistoricalCandleFetcher
                hist = HistoricalCandleFetcher()
                hist.interval = self.interval
                
                from core.utils import get_last_trading_day
                from datetime import datetime, timedelta
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                prev_date = get_last_trading_day(dt - timedelta(days=1)).strftime("%Y-%m-%d")
                
                prev_df = hist.fetch_single(instrument_key, "minutes", self.interval, prev_date, prev_date)
                if prev_df is not None and not prev_df.empty:
                    # Prepend only the very last candle of the previous day
                    df = pd.concat([prev_df.tail(1), df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    print(f"[IntradayFetcher] Anchored {instrument_key} to baseline from {prev_date}")
            except Exception as e:
                print(f"[IntradayFetcher] Warning: Baseline anchoring failed for {instrument_key}: {e}")

        return df

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        return self.get_candles(spot_key, date_str)
