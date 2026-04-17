"""Base class for all candle fetchers."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import upstox_client
from upstox_client.rest import ApiException

import core.config
from core.exceptions import ConfigurationError


class BaseCandleFetcher(ABC):
    """Abstract base; subclasses implement get_candles()."""

    def __init__(self, interval: int = core.config.CANDLE_INTERVAL_MINUTES):
        self.interval = interval
        token = core.config.UPSTOX_ACCESS_TOKEN
        if not token or (isinstance(token, str) and token.strip() in ("", "None")):
            raise ConfigurationError("UPSTOX_ACCESS_TOKEN not found in core.config.")
        
        cfg = upstox_client.Configuration()
        cfg.access_token = token
        self._api_client = upstox_client.ApiClient(cfg)
        self._history_api = upstox_client.HistoryV3Api(self._api_client)
        self.access_token = token
        self.used_fallback = False
        self.last_status   = None # To track HTTP status codes from Upstox
        self.last_error_code = None # To track specific Upstox error codes (e.g. UDAPI100050)

    @abstractmethod
    def get_candles(self, instrument_key: str, date_str: str, expiry_dt=None) -> pd.DataFrame | None:
        """Fetch option candles for a given day."""

    @abstractmethod
    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        """Fetch spot/underlying candles for a given day."""

    def warmup_cache(self, instrument_keys: list[str], date_str: str):
        """Optional hook to pre-fetch baseline data in parallel."""
        pass

    def _sync_token(self):
        """Ensures the API client uses the latest token from the live config."""
        current_token = core.config.UPSTOX_ACCESS_TOKEN
        if self._api_client.configuration.access_token != current_token:
            print(f"[Fetcher] Syncing Upstox token to latest broadcast version.")
            self._api_client.configuration.access_token = current_token
            self.access_token = current_token

    # ── Shared behavior ───────────────────────────────────────────────────────

    def _normalize_unit(self, timeframe: str) -> str:
        t = timeframe.lower().strip()
        aliases = {
            "minute":  "minutes",
            "hour":    "hours",
            "day":     "days",
            "week":    "weeks",
            "month":   "months",
        }
        return aliases.get(t, t)

    def _log_fetch(self, fetcher_name: str, instrument_key: str, from_date: str, to_date: str, interval: int | str, unit: str = "") -> None:
        """
        Centralised fetch log — prints a single clear line like:
          [Historical] NSE_FO|NIFTY... | 2026-04-10 → 2026-04-13 | 15-minutes
        """
        short_key = instrument_key.split("|")[-1]  # last segment is most readable
        interval_label = f"{interval}{('-' + unit) if unit else ''}"
        print(f"[{fetcher_name}] {short_key} | {from_date} -> {to_date} | {interval_label}")


    def _process_response(self, resp, date_str: str | None = None) -> pd.DataFrame | None:
        """
        Parse the Upstox candle response.
        Returns None if empty — caller's retry-loop handles holiday roll-over.
        Holidays are never cached; they are always confirmed live via API.
        """
        if not resp or not hasattr(resp, "data") or not resp.data or not resp.data.candles:
            return None
        # Upstox Historical API returns 6 columns (no OI), Intraday returns 7.
        first_candle = resp.data.candles[0]
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if len(first_candle) >= 7:
            cols.append("open_interest")

        df = pd.DataFrame(resp.data.candles, columns=cols)
        if "open_interest" not in df.columns:
            df["open_interest"] = 0
        df["date"] = pd.to_datetime(df["timestamp"])
        df.set_index("date", inplace=True)
        return df.sort_index()

    def _save_mock_response(self, resp, category: str, subpath: str):
        """Hidden: saves real API response to disk for mock data generation."""
        if not os.getenv("SAVE_MOCK_DATA"):
            return
        import json
        dir_path = os.path.join("tests", "mock_data", category)
        os.makedirs(dir_path, exist_ok=True)
        
        # Serialize response if not already a dict
        if hasattr(resp, "to_dict"):
            data = resp.to_dict()
        else:
            data = resp
            
        final_path = os.path.join(dir_path, f"{subpath}.json")
        with open(final_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
