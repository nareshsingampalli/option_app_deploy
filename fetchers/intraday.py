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
        return df

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        return self.get_candles(spot_key, date_str)
