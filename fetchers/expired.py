import upstox_client
from upstox_client.rest import ApiException
import pandas as pd

from core.rate_limiter import rate_limited
from core.config import UPSTOX_RATE_LIMIT_CALLS, UPSTOX_RATE_LIMIT_PERIOD
from fetchers.base import BaseCandleFetcher


class ExpiredCandleFetcher(BaseCandleFetcher):
    """SDK-based client for expired instrument endpoints."""

    def __init__(self, interval: int = 15):
        super().__init__(interval=interval)
        # Initialize the ExpiredInstrumentApi using the shared api_client from the base class
        self._api = upstox_client.ExpiredInstrumentApi(self._api_client)

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def fetch_expiries(self, underlying_key: str) -> list:
        self._sync_token()
        try:
            resp = self._api.get_expiries(underlying_key)
            if resp and hasattr(resp, "data"):
                self._save_mock_response(resp, "expired_expiries", underlying_key)
                return resp.data or []
            print(f"[ExpiredFetcher] fetch_expiries error: Invalid response format")
        except ApiException as e:
            print(f"[ExpiredFetcher] fetch_expiries error: {e}")
        except Exception as e:
            print(f"[ExpiredFetcher] fetch_expiries unexpected error: {e}")
        return []

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def fetch_contracts(self, underlying_key: str, expiry_date: str) -> list:
        self._sync_token()
        try:
            # We default to option contracts, but future contracts are also available in the SDK
            resp = self._api.get_expired_option_contracts(underlying_key, expiry_date)
            if resp and hasattr(resp, "data"):
                self._save_mock_response(resp, "expired_contracts", f"{underlying_key}_{expiry_date}")
                return resp.data or []
            print(f"[ExpiredFetcher] fetch_contracts error: Invalid response format")
        except ApiException as e:
            print(f"[ExpiredFetcher] fetch_contracts error: {e}")
        except Exception as e:
            print(f"[ExpiredFetcher] fetch_contracts unexpected error: {e}")
        return []

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def fetch_candle_data(
        self, instrument_key: str, interval_str: str, to_date: str, from_date: str
    ) -> pd.DataFrame | None:
        self._sync_token()
        try:
            resp = self._api.get_expired_historical_candle_data(
                instrument_key, interval_str, to_date, from_date
            )
            if resp:
                self._save_mock_response(resp, "expired", instrument_key)
                return self._process_response(resp)
            print(f"[ExpiredFetcher] fetch_candle_data error: Invalid response format")
        except ApiException as e:
            print(f"[ExpiredFetcher] fetch_candle_data error: {e}")
        except Exception as e:
            print(f"[ExpiredFetcher] fetch_candle_data unexpected error: {e}")
        return None

    # ── Unified interface ────────────────────────────────────────────────────
    def _get_prev_trading_day(self, date_str: str) -> str:
        """Helper to find the preceding trading session date, skipping weekends & holidays."""
        from datetime import datetime, timedelta
        from core.utils import get_last_trading_day
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        prev_dt = dt - timedelta(days=1)
        return get_last_trading_day(prev_dt).strftime("%Y-%m-%d")

    def get_candles(self, instrument_key: str, date_str: str, expiry_dt=None) -> pd.DataFrame | None:
        """Fetch expired candles from prev trading day through target date for OI ROC baseline continuity."""
        key = instrument_key
        if key.count("|") < 2 and expiry_dt is not None:
            key = f"{key}|{expiry_dt.strftime('%d-%m-%Y')}"

        interval_str = f"{self.interval}minute"
        # Fetch from previous trading day to provide OI ROC baseline (same as HistoricalCandleFetcher)
        from_date = self._get_prev_trading_day(date_str)
        return self.fetch_candle_data(key, interval_str, date_str, from_date)

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        """Spot/index data is NOT available via expired API — callers should use HistoricalFetcher."""
        raise NotImplementedError(
            "Spot index data is unavailable via the expired API. "
            "Use HistoricalCandleFetcher for spot candles."
        )
