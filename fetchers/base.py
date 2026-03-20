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
        self.last_status   = None # To track HTTP status codes from Upstox (e.g. 404 for holiday)

    @abstractmethod
    def get_candles(self, instrument_key: str, date_str: str, expiry_dt=None) -> pd.DataFrame | None:
        """Fetch option candles for a given day."""

    @abstractmethod
    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        """Fetch spot/underlying candles for a given day."""

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

    def _process_response(self, resp) -> pd.DataFrame | None:
        if not resp or not hasattr(resp, "data") or not resp.data or not resp.data.candles:
            return None
        df = pd.DataFrame(
            resp.data.candles,
            columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        )
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
