"""Background market-hours schedulers (one thread per exchange)."""

from __future__ import annotations

import sys
import subprocess
import threading
import time
from datetime import datetime, timedelta

from dashboard import app, socketio
from core.config import SCHEDULER_HOURS, CANDLE_INTERVAL_MINUTES
from core.utils import ist_now

# Per-symbol fetch locks — dynamic
_locks_lock = threading.Lock()
_fetch_locks: dict[str, threading.Lock] = {}
_wake_event = threading.Event()

def wake_scheduler():
    _wake_event.set()

def get_lock(symbol: str) -> threading.Lock:
    with _locks_lock:
        if symbol not in _fetch_locks:
            _fetch_locks[symbol] = threading.Lock()
        return _fetch_locks[symbol]

def _secs(t) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second

from storage.db_storage import build_storage_chain
from strategies.handlers import build_pipeline

def _run_fetch(symbol: str, exchange_prefix: str):
    from core.utils import get_last_trading_day
    today = get_last_trading_day().strftime("%Y-%m-%d")
    lock  = get_lock(symbol)
    if not lock.acquire(blocking=False):
        return
    try:
        print(f"[Scheduler] Fetching {symbol} ({exchange_prefix}) live data...")
        storage  = build_storage_chain()
        pipeline = build_pipeline(exchange_prefix, today, True, symbol, storage)

        if pipeline:
            pipeline.run(symbol, today, None)
        
        with app.app_context():
            from dashboard import socketio
            # Emit to the room specifically for this symbol
            socketio.emit("data_updated", {"symbol": symbol, "prefix": exchange_prefix}, room=symbol)
            # Legacy support for the exchange-wide prefix (optional)
            socketio.emit("data_updated", {"prefix": exchange_prefix}, room=exchange_prefix)
            
    except Exception as e:
        print(f"[Scheduler-{symbol}] Error: {e}")
    finally:
        lock.release()

def _main_scheduler_loop():
    """Single loop that manages all active exchange subscriptions."""
    from dashboard.routes import get_active_symbols
    from core.config import NSE_INDEX_KEYS, MCX_FUT_KEYS
    
    print("[Scheduler] Dynamic Master Loop started.")
    last_fetch_times = {} # symbol -> last_min
    last_delay_min = -1   # globally track delay for this minute cycle

    while True:
        now = ist_now()
        cur_min = now.minute
        active_symbols = get_active_symbols()
        
        # Only proceed if today is a trading day (skip weekends and holidays)
        from core.utils import get_last_trading_day
        is_trading_day = (get_last_trading_day(now).strftime("%Y-%m-%d") == now.strftime("%Y-%m-%d"))

        if is_trading_day:
            for symbol in active_symbols:
                if not symbol: continue
                sym_upper = symbol.upper()
                
                # Determine exchange and config
                if sym_upper in NSE_INDEX_KEYS or sym_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                    cfg = SCHEDULER_HOURS["NSE"]
                    prefix = "NSE"
                elif sym_upper in MCX_FUT_KEYS or sym_upper in ["CRUDEOIL", "NATURALGAS", "SILVER", "GOLD"]:
                    cfg = SCHEDULER_HOURS["MCX"]
                    prefix = "MCX"
                elif sym_upper in ["SENSEX", "BANKEX"]:
                    cfg = SCHEDULER_HOURS["BSE"]
                    prefix = "BSE"
                else: continue

                # Check market hours
                start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
                end_t   = datetime.strptime(cfg["end"],   "%H:%M:%S").time()
                
                if _secs(start_t) <= _secs(now.time()) <= _secs(end_t):
                    # Centralized interval cycle
                    is_startup = (now.hour == 9 and 16 <= now.minute <= 20)
                    is_regular = (now.minute % CANDLE_INTERVAL_MINUTES == 0)

                    if (is_startup or is_regular) and last_fetch_times.get(symbol) != cur_min:
                        # Stagger burst once per minute cycle
                        if last_delay_min != cur_min:
                            delay = 10  # default: wait for Upstox to finalize candle
                            if now.minute % 5 == 0: delay = 30  # 5-min congestion avoidance
                            
                            print(f"[Scheduler] Minute {now.minute}: pausing {delay}s before burst...")
                            time.sleep(delay)
                            last_delay_min = cur_min

                        threading.Thread(
                            target=_run_fetch, 
                            args=(symbol, prefix),
                            daemon=True
                        ).start()
                        last_fetch_times[symbol] = cur_min
        
        # Check if any symbol was processed (in market hours)
        if not active_symbols:
            # No clients watching anything
            _wake_event.wait(timeout=60)
            _wake_event.clear()
            continue

        any_in_hours = False
        from core.utils import get_last_trading_day
        is_trading_day = (get_last_trading_day(now).strftime("%Y-%m-%d") == now.strftime("%Y-%m-%d"))

        if is_trading_day:
            for symbol in active_symbols:
                if not symbol: continue
                sym_upper = symbol.upper()
                if sym_upper in NSE_INDEX_KEYS or sym_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                    cfg = SCHEDULER_HOURS["NSE"]
                elif sym_upper in MCX_FUT_KEYS or sym_upper in ["CRUDEOIL", "NATURALGAS", "SILVER", "GOLD"]:
                    cfg = SCHEDULER_HOURS["MCX"]
                elif sym_upper in ["SENSEX", "BANKEX"]:
                    cfg = SCHEDULER_HOURS["BSE"]
                else: continue
                
                start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
                end_t   = datetime.strptime(cfg["end"],   "%H:%M:%S").time()
                if _secs(start_t) <= _secs(now.time()) <= _secs(end_t):
                    any_in_hours = True
                    break
        
        if not any_in_hours:
            # Market is closed for all watched symbols. 
            # Kill the loop until the next morning (9:00 AM) or until a user connects.
            next_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if now > next_open:
                next_open += timedelta(days=1)
            sleep_sec = (next_open - now).total_seconds()
            
            print(f"[Scheduler] Market is closed. Hibernating completely for {sleep_sec/3600:.1f} hours till next open...")
            _wake_event.wait(timeout=sleep_sec)
            _wake_event.clear()
        else:
            _wake_event.wait(timeout=20)
            _wake_event.clear()

def start_schedulers():
    t = threading.Thread(target=_main_scheduler_loop, daemon=True, name="MasterScheduler")
    t.start()
