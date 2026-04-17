"""
Strategy Chain of Responsibility
---------------------------------
Determines the correct pipeline for a given (exchange, date, mode) context.

Chain order
───────────────────────────────────────────────────────────────────────────
LiveHandler         → live_mode == True                → Intraday pipeline
TodayGuardHandler   → date == today and not live       → Block (no data)
ExpiredHandler      → NSE + date <= last_expired_dt    → Expired pipeline
HistoricalHandler   → all other past dates             → Historical pipeline
FutureGuardHandler  → date > today                     → Block (no data)
───────────────────────────────────────────────────────────────────────────

Usage
-----
    from strategies.handlers import StrategyChain, StrategyContext

    ctx      = StrategyContext(exchange="NSE", target_date="2026-02-10", live_mode=False)
    pipeline = StrategyChain.build("NSE", storage_chain).resolve(ctx)
    pipeline.run("NIFTY", ctx.target_date)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import pytz
from typing import Optional

from fetchers.factory import CandleFetcherFactory
from fetchers.expired import ExpiredCandleFetcher
from resolvers.nse_resolver import NSEActiveResolver, NSEExpiredResolver
from resolvers.mcx_resolver import MCXInstrumentResolver
from resolvers.bse_resolver import BSEActiveResolver
from storage.base import StorageHandler
from strategies.base import MarketDataPipeline


@dataclass
class StrategyContext:
    exchange:        str
    target_date:     str
    live_mode:       bool               = False
    symbol:          str                = "NIFTY"
    last_expired_dt: Optional[datetime] = field(default=None)
    today_str:       str                = field(default_factory=lambda: datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d"))
    interval:        int                = 15
    expiry_offset:   int                = 0


# ── Abstract Handler ──────────────────────────────────────────────────────────

class StrategyHandler(ABC):
    def __init__(self):
        self._next: Optional[StrategyHandler] = None

    def set_next(self, h: StrategyHandler) -> StrategyHandler:
        self._next = h
        return h

    @abstractmethod
    def handle(self, ctx: StrategyContext, storage: StorageHandler) -> Optional[MarketDataPipeline]:
        pass

    def _forward(self, ctx: StrategyContext, storage: StorageHandler) -> Optional[MarketDataPipeline]:
        return self._next.handle(ctx, storage) if self._next else None


# ── Concrete Handlers ────────────────────────────────────────────────────────

class LiveHandler(StrategyHandler):
    def handle(self, ctx, storage):
        if not ctx.live_mode:
            return self._forward(ctx, storage)
        fetcher = CandleFetcherFactory.create(ctx.target_date, live_mode=True, interval=ctx.interval)
        print(f"[StrategyChain] LIVE -> {ctx.exchange}")
        if ctx.exchange == "MCX":
            from strategies.mcx import MCXLivePipeline
            return MCXLivePipeline(fetcher, MCXInstrumentResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)
        if ctx.exchange == "BSE":
            from strategies.bse import BSELivePipeline
            return BSELivePipeline(fetcher, BSEActiveResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)
        from strategies.nse import NSELivePipeline
        return NSELivePipeline(fetcher, NSEActiveResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)


class TodayGuardHandler(StrategyHandler):
    """
    Routes today's requests through the intraday-capable pipeline, but ONLY
    when live_mode is True (i.e. the user explicitly wants live broker data).

    When live_mode is False and date == today (e.g. user toggled Live OFF or
    viewing today historically after market close), we fall through to
    HistoricalHandler so cached / historical-API data is served without any
    live broker call.
    """
    def handle(self, ctx, storage):
        # RULE: If the user is viewing TODAY's date, we MUST use the Intraday-capable
        # pipeline, regardless of whether the LIVE toggle is on or off. 
        # Why? Because the Historical API (/historical) does not contain candles
        # for the current date until after the midnight maintenance sync.
        if ctx.target_date == ctx.today_str:
            print(f"[StrategyChain] TODAY -> {ctx.exchange} (Forced Intraday Pipeline for current date)")
            fetcher = CandleFetcherFactory.create(ctx.target_date, live_mode=True, interval=ctx.interval)

            if ctx.exchange == "MCX":
                from strategies.mcx import MCXLivePipeline
                return MCXLivePipeline(fetcher, MCXInstrumentResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)
            if ctx.exchange == "BSE":
                from strategies.bse import BSELivePipeline
                return BSELivePipeline(fetcher, BSEActiveResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)
            from strategies.nse import NSELivePipeline
            return NSELivePipeline(fetcher, NSEActiveResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)

        return self._forward(ctx, storage)




class ExpiredHandler(StrategyHandler):
    """
    NSE-only: routes to ExpiredPipeline ONLY when the selected date is
    explicitly within the expired bucket (target_date <= last_expired_dt
    returned by the Upstox expired-instruments API).

    Rule:
      - target_date <= last_expired_dt  → ExpiredCandleFetcher
      - anything else                   → falls through to HistoricalCandleFetcher

    No fallback resolution against the active series is performed.
    Holidays are handled naturally by HistoricalCandleFetcher: Upstox's
    historical API returns the nearest available trading day in the requested
    date range, so no explicit holiday rollback is needed here.
    """
    def handle(self, ctx, storage):
        if ctx.exchange != "NSE":
            return self._forward(ctx, storage)

        target_dt = datetime.strptime(ctx.target_date, "%Y-%m-%d")
        is_expired = (
            ctx.last_expired_dt is not None
            and target_dt <= ctx.last_expired_dt
        )

        if is_expired:
            print(f"[StrategyChain] EXPIRED ({ctx.target_date} <= {ctx.last_expired_dt.date()}) → ExpiredCandleFetcher")
            fetcher = CandleFetcherFactory.create(
                ctx.target_date, last_expired_dt=target_dt, interval=ctx.interval
            )
            from strategies.nse import NSEExpiredPipeline
            from resolvers.nse_resolver import NSEExpiredResolver
            return NSEExpiredPipeline(fetcher, NSEExpiredResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)

        return self._forward(ctx, storage)



class HistoricalHandler(StrategyHandler):
    def handle(self, ctx, storage):
        target_dt = datetime.strptime(ctx.target_date, "%Y-%m-%d")
        today_dt  = datetime.strptime(ctx.today_str,   "%Y-%m-%d")
        if target_dt > today_dt:
            return self._forward(ctx, storage)
        fetcher = CandleFetcherFactory.create(ctx.target_date, interval=ctx.interval)
        print(f"[StrategyChain] HISTORICAL -> {ctx.exchange}")
        if ctx.exchange == "MCX":
            from strategies.mcx import MCXHistoricalPipeline
            return MCXHistoricalPipeline(fetcher, MCXInstrumentResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)
        if ctx.exchange == "BSE":
            from strategies.bse import BSEHistoricalPipeline
            return BSEHistoricalPipeline(fetcher, BSEActiveResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)
        from strategies.nse import NSEHistoricalPipeline
        return NSEHistoricalPipeline(fetcher, NSEActiveResolver(), storage, symbol=ctx.symbol, expiry_offset=ctx.expiry_offset)


class FutureGuardHandler(StrategyHandler):
    def handle(self, ctx, storage):
        print(f"[StrategyChain] WARNING: {ctx.target_date} is in the future. No data.")
        return None


# ── StrategyChain builder ─────────────────────────────────────────────────────

class StrategyChain:
    """Builds the handler chain and exposes a single resolve() entry point."""

    @staticmethod
    def build(storage: StorageHandler) -> StrategyChain:
        live    = LiveHandler()
        guard   = TodayGuardHandler()
        expired = ExpiredHandler()
        hist    = HistoricalHandler()
        future  = FutureGuardHandler()

        live.set_next(guard).set_next(expired).set_next(hist).set_next(future)
        return StrategyChain(live)

    def __init__(self, head: StrategyHandler):
        self._head = head

    def resolve(self, ctx: StrategyContext, storage: StorageHandler) -> Optional[MarketDataPipeline]:
        return self._head.handle(ctx, storage)


def _fetch_last_expired_dt(symbol: str) -> Optional[datetime]:
    """Query Upstox expired expiries API and return the most recent date."""
    from core.config import NSE_INDEX_KEYS, BSE_INDEX_KEYS
    try:
        if symbol.upper() in BSE_INDEX_KEYS:
            underlying = BSE_INDEX_KEYS[symbol.upper()]
        else:
            underlying = NSE_INDEX_KEYS.get(symbol.upper(), NSE_INDEX_KEYS["NIFTY"])
        fetcher    = ExpiredCandleFetcher()
        expiries   = fetcher.fetch_expiries(underlying)
        if expiries:
            return datetime.strptime(str(sorted(expiries)[-1]), "%Y-%m-%d")
    except Exception as e:
        print(f"[StrategyChain] Could not fetch expired expiries: {e}")
    return None


def build_pipeline(
    exchange:    str,
    target_date: str,
    live_mode:   bool,
    symbol:      str,
    storage:     StorageHandler,
    interval:    int = 15,
    next_expiry: bool = False
) -> Optional[MarketDataPipeline]:
    """
    Convenience function — resolves last_expired_dt and returns the correct pipeline.

    Args:
        exchange    : 'NSE' or 'MCX'
        target_date : 'YYYY-MM-DD'
        live_mode   : True if --live flag set
        symbol      : 'NIFTY', 'BANKNIFTY', 'CRUDEOIL', etc.
        storage     : configured StorageHandler chain
    """
    # Force BSE if it's Sensex/Bankex, regardless of what frontend says
    if symbol.upper() in ["SENSEX", "BANKEX"]:
        exchange = "BSE"

    print(f"[StrategyChain] Resolving pipeline for {symbol} on {target_date} (live={live_mode}, exchange={exchange})...")
    
    # ── Safety Check: Live mode is ONLY for today ────────────────────────────
    from core.utils import ist_now
    today_str = ist_now().strftime("%Y-%m-%d")
    if live_mode and target_date != today_str:
        print(f"[StrategyChain] Correcting live_mode -> False (reason: {target_date} != today)")
        live_mode = False

    last_expired = None
    if exchange == "NSE" and not live_mode:
        print(f"[StrategyChain] Fetching last expired date for NSE...")
        last_expired = _fetch_last_expired_dt(symbol)
        print(f"[StrategyChain] Last expired date: {last_expired}")
    
    ctx = StrategyContext(
        exchange=exchange,
        target_date=target_date,
        live_mode=live_mode,
        symbol=symbol,
        last_expired_dt=last_expired,
        interval=interval,
        expiry_offset=1 if next_expiry else 0
    )
    chain = StrategyChain.build(storage)
    return chain.resolve(ctx, storage)
