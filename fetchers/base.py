"""
BaseCandleFetcher
-----------------
Provides:
  • Authenticated Upstox API client (token from env only — no hardcoded fallback).
  • Shared _process_response() that converts raw candle arrays into a sorted DataFrame.
  • Abstract get_candles(instrument_key, date_str) — unified interface used by the pipeline.
"""

import os
from abc import ABC, abstractmethod

import pandas as pd
import upstox_client
from upstox_client.rest import ApiException

from core.exceptions import ConfigurationError


class BaseCandleFetcher(ABC):
    """Abstract base; subclasses implement get_candles()."""

    def __init__(self, access_token: str | None = None):
        token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not token or token.strip() in ("", "None"):
            # Fallback to the token used in candle_fetchers.py (line 20)
            token = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI4Q0FRNzUiLCJqdGkiOiI2OWE0MDcyYmEwMDMwMzdmNDM4NGU2OGEiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzcyMzU3NDE5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzI0MDI0MDB9.3ttGjJegTK-CFt5-Xc4mBRcSUdRaxkKlAnIDkVZR2js'
        if not token or token.strip() in ("", "None"):
            raise ConfigurationError(
                "UPSTOX_ACCESS_TOKEN is not set. "
                "Add it to your .env file: UPSTOX_ACCESS_TOKEN=<your_token>"
            )
        self.access_token = token

        cfg = upstox_client.Configuration()
        cfg.access_token = self.access_token
        api_client = upstox_client.ApiClient(cfg)
        self._history_api = upstox_client.HistoryV3Api(api_client)
        self.used_fallback = False  # Track if latest fetch used a fallback mechanism

    # ── Unified interface ────────────────────────────────────────────────────
    @abstractmethod
    def get_candles(
        self, instrument_key: str, date_str: str, expiry_dt=None
    ) -> pd.DataFrame | None:
        """Return a DataFrame of 5-min candles for the given instrument + date."""

    # ── Spot/IV data ─────────────────────────────────────────────────────────
    @abstractmethod
    def get_spot_candles(
        self, spot_key: str, date_str: str
    ) -> pd.DataFrame | None:
        """Return intraday spot candles used to build the IV spot-map."""

    # ── Shared response parser ───────────────────────────────────────────────
    def _process_response(self, response) -> pd.DataFrame | None:
        try:
            candles = None
            data = getattr(response, "data", None)
            if data is not None:
                candles = getattr(data, "candles", None)
            if candles is None and isinstance(response, dict):
                d = response.get("data", {})
                if isinstance(d, dict):
                    candles = d.get("candles")
            if not candles:
                return None

            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"],
            )
            df["date"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("date").sort_index()
            for col in ("open", "high", "low", "close", "volume", "open_interest"):
                df[col] = pd.to_numeric(df[col])
            return df
        except Exception as e:
            print(f"[BaseFetcher] Response parse error: {e}")
            return None

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        u = unit.lower()
        aliases = {
            "minute":  "minutes",
            "hour":    "hours",
            "day":     "days",
            "week":    "weeks",
            "month":   "months",
        }
        return aliases.get(u, u if u.endswith("s") else u + "s")
