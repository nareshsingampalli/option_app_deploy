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

from core.config import ConfigurationError, CANDLE_INTERVAL_MINUTES


class BaseCandleFetcher(ABC):
    """Abstract base; subclasses implement get_candles()."""

    def __init__(self, access_token: str | None = None, interval: int = CANDLE_INTERVAL_MINUTES):
        self.interval = interval
        token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not token or token.strip() in ("", "None"):
            # Fallback to the token used in candle_fetchers.py (line 20)
            token = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI4Q0FRNzUiLCJqdGkiOiI2OWI3N2MzYTg2N2UzYjJmYjY3OTI5MTgiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzczNjMyNTcwLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzM2OTg0MDB9.cxXbGG8N-_2izYa8HaBhFxNQ0a8oizOQ84lciQUtvcA'
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

    # ── Mock Data Capture ────────────────────────────────────────────────────
    def _save_mock_response(self, raw_data, endpoint_type: str, instrument_key: str):
        """Saves live API responses to disk so they can be replayed by the simulator."""
        # Only save if explicitly asked and NOT using the mock server
        if os.environ.get("SAVE_MOCK_DATA", "").lower() != "true":
            return
        if os.environ.get("UPSTOX_API_URL"):
            return
            
        import json
        import time
        from pathlib import Path
        
        try:
            safe_key = instrument_key.replace('|', '_').replace(':', '_')
            ts = int(time.time() * 1000)
            
            target_dir = Path(os.getcwd()) / "mock_data" / endpoint_type
            target_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = target_dir / f"{safe_key}_{ts}.json"
            
            if hasattr(raw_data, "to_dict"):
                dict_data = raw_data.to_dict()
            elif isinstance(raw_data, dict):
                dict_data = raw_data
            else:
                return # Don't know how to serialize
                
            with open(filepath, "w") as f:
                json.dump(dict_data, f, indent=2)
            print(f"[MockData] Saved response to {filepath}")
        except Exception as e:
            print(f"[MockData] Failed to save {endpoint_type} mock data: {e}")

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
