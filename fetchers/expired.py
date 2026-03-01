"""ExpiredCandleFetcher — Upstox /v2/expired-instruments/* REST endpoints."""

import urllib.parse
import requests
import pandas as pd

from core.rate_limiter import rate_limited
from core.config import UPSTOX_RATE_LIMIT_CALLS, UPSTOX_RATE_LIMIT_PERIOD
from fetchers.base import BaseCandleFetcher

_BASE = "https://api.upstox.com/v2/expired-instruments"


class ExpiredCandleFetcher(BaseCandleFetcher):
    """REST client for expired instrument endpoints."""

    def __init__(self, access_token: str | None = None):
        super().__init__(access_token)
        self._headers = {
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def fetch_expiries(self, underlying_key: str) -> list:
        safe = urllib.parse.quote(underlying_key)
        url  = f"{_BASE}/expiries?instrument_key={safe}"
        try:
            r = requests.get(url, headers=self._headers, timeout=15)
            if r.status_code == 200:
                return r.json().get("data", [])
            print(f"[ExpiredFetcher] fetch_expiries {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[ExpiredFetcher] fetch_expiries error: {e}")
        return []

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def fetch_contracts(self, underlying_key: str, expiry_date: str) -> list:
        safe = urllib.parse.quote(underlying_key)
        url  = f"{_BASE}/option/contract?instrument_key={safe}&expiry_date={expiry_date}"
        try:
            r = requests.get(url, headers=self._headers, timeout=15)
            if r.status_code == 200:
                return r.json().get("data", [])
            print(f"[ExpiredFetcher] fetch_contracts {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[ExpiredFetcher] fetch_contracts error: {e}")
        return []

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def fetch_candle_data(
        self, instrument_key: str, interval_str: str, to_date: str, from_date: str
    ) -> pd.DataFrame | None:
        safe = urllib.parse.quote(instrument_key)
        url  = f"{_BASE}/historical-candle/{safe}/{interval_str}/{to_date}/{from_date}"
        try:
            r = requests.get(url, headers=self._headers, timeout=15)
            if r.status_code == 200:
                return self._process_response(r.json())
            print(f"[ExpiredFetcher] fetch_candle_data {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[ExpiredFetcher] fetch_candle_data error: {e}")
        return None

    # ── Unified interface ────────────────────────────────────────────────────
    def get_candles(self, instrument_key: str, date_str: str, expiry_dt=None) -> pd.DataFrame | None:
        """Appends expiry suffix to key if missing, then fetches 5-min candles."""
        key = instrument_key
        if key.count("|") < 2 and expiry_dt is not None:
            key = f"{key}|{expiry_dt.strftime('%d-%m-%Y')}"
        print(f"[ExpiredFetcher] Fetching expired candles: key={key}, date={date_str}")
        return self.fetch_candle_data(key, "5minute", date_str, date_str)

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        """Spot/index data is NOT available via expired API — callers should use HistoricalFetcher."""
        raise NotImplementedError(
            "Spot index data is unavailable via the expired API. "
            "Use HistoricalCandleFetcher for spot candles."
        )
