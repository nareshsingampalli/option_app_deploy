"""
CandleFetcherFactory
--------------------
Selects the correct BaseCandleFetcher subclass based on:
  - exchange  ('NSE' or 'MCX')
  - live_mode (bool)
  - target_date vs last_expired_dt (NSE only)

Decision table
──────────────────────────────────────────────────────────
live_mode = True              → IntradayCandleFetcher
NSE + date ≤ last_expired_dt  → ExpiredCandleFetcher
anything else                 → HistoricalCandleFetcher
──────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime

from fetchers.base import BaseCandleFetcher
from fetchers.intraday import IntradayCandleFetcher
from fetchers.historical import HistoricalCandleFetcher
from fetchers.expired import ExpiredCandleFetcher


class CandleFetcherFactory:
    """Factory — produce the correct fetcher for a given context."""

    @staticmethod
    def create(
        target_date: str,
        live_mode: bool = False,
        last_expired_dt: datetime | None = None,
    ) -> BaseCandleFetcher:
        """
        Parameters
        ----------
        target_date  : 'YYYY-MM-DD'
        live_mode    : True when --live flag is active
        last_expired_dt : Most recent expired-contract datetime (optional)

        Returns
        -------
        An instantiated BaseCandleFetcher subclass.
        """
        if live_mode:
            print("[FetcherFactory] LIVE -> IntradayCandleFetcher")
            return IntradayCandleFetcher()

        if last_expired_dt is not None:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            if target_dt <= last_expired_dt:
                print(f"[FetcherFactory] EXPIRED -> ExpiredCandleFetcher ({target_date})")
                return ExpiredCandleFetcher()

        print(f"[FetcherFactory] HIST -> HistoricalCandleFetcher ({target_date})")
        return HistoricalCandleFetcher()
