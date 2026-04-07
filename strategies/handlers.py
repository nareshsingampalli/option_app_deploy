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
            return MCXLivePipeline(fetcher, MCXInstrumentResolver(), storage, symbol=ctx.symbol)
        if ctx.exchange == "BSE":
            from strategies.bse import BSELivePipeline
            return BSELivePipeline(fetcher, BSEActiveResolver(), storage, symbol=ctx.symbol)
        from strategies.nse import NSELivePipeline
        return NSELivePipeline(fetcher, NSEActiveResolver(), storage, symbol=ctx.symbol)


class TodayGuardHandler(StrategyHandler):
    def handle(self, ctx, storage):
        # Block today's date if not in Live mode OR market is currently open
        if ctx.target_date == ctx.today_str and not ctx.live_mode:
            from core.utils import ist_now
            from core.config import SCHEDULER_HOURS
            
            # Check if market is actually open right now for this exchange
            cfg = SCHEDULER_HOURS.get(ctx.exchange, SCHEDULER_HOURS["NSE"])
            now_t = ist_now().time()
            start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
            end_t = datetime.strptime(cfg["end"], "%H:%M:%S").time()
            
            # Convert to seconds for comparison
            def s(t): return t.hour * 3600 + t.minute * 60 + t.second
            
            is_open = s(start_t) <= s(now_t) <= s(end_t)
            
            if is_open:
                print(f"[StrategyChain] BLOCKED: {ctx.target_date} is today and market is OPEN. Enable Live mode.")
                return None
            else:
                print(f"[StrategyChain] ALLOWED: {ctx.target_date} is today but market is CLOSED. Accessing historical snapshot.")
                # Allow it to fall through to HistoricalHandler
            
        return self._forward(ctx, storage)


class ExpiredHandler(StrategyHandler):
    """NSE-only: routes to ExpiredPipeline when date <= last_expired_dt."""
    def handle(self, ctx, storage):
        if ctx.exchange != "NSE":
            return self._forward(ctx, storage)
        if ctx.last_expired_dt is None:
            return self._forward(ctx, storage)
        target_dt = datetime.strptime(ctx.target_date, "%Y-%m-%d")
        if target_dt <= ctx.last_expired_dt:
            print(f"[StrategyChain] EXPIRED ({ctx.target_date} <= {ctx.last_expired_dt.date()})")
            fetcher = CandleFetcherFactory.create(
                ctx.target_date, last_expired_dt=ctx.last_expired_dt, interval=ctx.interval
            )
            from strategies.nse import NSEExpiredPipeline
            return NSEExpiredPipeline(fetcher, NSEExpiredResolver(), storage, symbol=ctx.symbol)
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
            return MCXHistoricalPipeline(fetcher, MCXInstrumentResolver(), storage, symbol=ctx.symbol)
        if ctx.exchange == "BSE":
            from strategies.bse import BSEHistoricalPipeline
            return BSEHistoricalPipeline(fetcher, BSEActiveResolver(), storage, symbol=ctx.symbol)
        from strategies.nse import NSEHistoricalPipeline
        return NSEHistoricalPipeline(fetcher, NSEActiveResolver(), storage, symbol=ctx.symbol)


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
    interval:    int = 15
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
    )
    chain = StrategyChain.build(storage)
    return chain.resolve(ctx, storage)
