"""
CandleFetcherFactory
--------------------
Selects the correct BaseCandleFetcher subclass based on:
  - exchange  ('NSE' or 'MCX')
  - live_mode (bool)
  - target_date vs last_expired_dt (NSE only)

Decision table
----------------------------------------------------------
live_mode = True              -> IntradayCandleFetcher
NSE + date <= last_expired_dt  -> ExpiredCandleFetcher
anything else                 -> HistoricalCandleFetcher
----------------------------------------------------------
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
        interval: int = 15
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
        # RULE: If date is today, always use Intraday fetcher (snapshot or live).
        # /historical API is empty for the current calendar date.
        from core.utils import ist_now
        today_str = ist_now().strftime("%Y-%m-%d")
        
        if live_mode or target_date == today_str:
            print(f"[FetcherFactory] {target_date} == {today_str} -> IntradayCandleFetcher")
            return IntradayCandleFetcher(interval=interval)


        if last_expired_dt is not None:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            if target_dt <= last_expired_dt:
                print(f"[FetcherFactory] EXPIRED -> ExpiredCandleFetcher ({target_date}, int={interval})")
                return ExpiredCandleFetcher(interval=interval)

        print(f"[FetcherFactory] HIST -> HistoricalCandleFetcher ({target_date}, int={interval})")
        return HistoricalCandleFetcher(interval=interval)
