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

def _run_fetch(symbol: str, exchange_prefix: str, interval: int = 15, next_expiry: bool = False):
    from core.utils import ist_now
    today = ist_now().strftime("%Y-%m-%d")
    # Lock is per-symbol, per-interval, AND per-expiry track
    lock_key = f"{symbol}_{interval}_{'next' if next_expiry else 'cur'}"
    lock = get_lock(lock_key)
    
    if not lock.acquire(blocking=False):
        return
    try:
        suffix = " (Next Expiry)" if next_expiry else ""
        print(f"[Scheduler] Fetching {symbol}{suffix} at {interval}m interval...")
        storage  = build_storage_chain()
        pipeline = build_pipeline(exchange_prefix, today, True, symbol, storage, interval=interval, next_expiry=next_expiry)

        if pipeline:
            pipeline.run(symbol, today, None)
            
            with app.app_context():
                socketio.emit("data_updated", {
                    "symbol": symbol, 
                    "prefix": exchange_prefix, 
                    "interval": interval,
                    "next_expiry": next_expiry
                }, room=symbol)
                
    except Exception as e:
        print(f"[Scheduler-{symbol}-{interval}-next{next_expiry}] Error: {e}")
    finally:
        lock.release()

def _main_scheduler_loop():
    """Single loop that manages all active exchange subscriptions."""
    from dashboard.routes import get_active_symbols
    from core.config import NSE_INDEX_KEYS, MCX_FUT_KEYS
    
    print("[Scheduler] Dynamic Master Loop started.")
    last_fetch_times = {} # key -> last_min_hour

    while True:
        now = ist_now()
        cur_min = now.minute
        cur_hour = now.hour
        # active_map is now { 'NIFTY': {(15, False), (15, True)}, ... }
        active_map = get_active_symbols()
        
        is_trading_day = now.weekday() < 5

        if is_trading_day and active_map:
            for symbol, tracks in active_map.items():
                if not symbol: continue
                sym_upper = symbol.upper()
                
                # ... [Exchange Resolve Logic same as before] ...
                if sym_upper in NSE_INDEX_KEYS or sym_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                    cfg, prefix = SCHEDULER_HOURS["NSE"], "NSE"
                elif sym_upper in MCX_FUT_KEYS or sym_upper in ["CRUDEOIL", "NATURALGAS", "SILVER", "GOLD"]:
                    cfg, prefix = SCHEDULER_HOURS["MCX"], "MCX"
                elif sym_upper in ["SENSEX", "BANKEX"]:
                    cfg, prefix = SCHEDULER_HOURS["BSE"], "BSE"
                else: continue

                start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
                end_t   = datetime.strptime(cfg["end"],   "%H:%M:%S").time()
                
                if _secs(start_t) <= _secs(now.time()) <= _secs(end_t):
                    for (interval, next_exp) in tracks:
                        fetch_key = f"{symbol}_{interval}_{next_exp}"
                        
                        is_startup = (cur_hour == 9 and 16 <= cur_min <= 20)
                        
                        if interval == 1:
                            is_due = (cur_min % 2 == 0)
                        else:
                            is_due = (cur_min % interval == 0)

                        if (is_startup or is_due) and last_fetch_times.get(fetch_key) != (cur_hour, cur_min):
                            delay = 2 if cur_min % 5 != 0 else 5
                            
                            def delayed_start(s, p, i, n, d):
                                time.sleep(d)
                                _run_fetch(s, p, i, n)

                            threading.Thread(
                                target=delayed_start, 
                                args=(symbol, prefix, interval, next_exp, delay),
                                daemon=True
                            ).start()
                            
                            last_fetch_times[fetch_key] = (cur_hour, cur_min)
        
        # Check if any symbol was processed (in market hours)
        if not active_map:
            # No clients watching anything
            _wake_event.wait(timeout=60)
            _wake_event.clear()
            continue

        any_in_hours = False
        # Simple trading day check: exclude weekends. Holidays handled by empty fetch responses.
        is_trading_day = now.weekday() < 5
        
        if is_trading_day:
            for symbol in active_map.keys():
                if not symbol: continue
                sym_upper = symbol.upper()
                if sym_upper in NSE_INDEX_KEYS or sym_upper in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                    cfg = SCHEDULER_HOURS["NSE"]
                elif sym_upper in MCX_FUT_KEYS or sym_upper in ["CRUDEOIL", "NATURALGAS", "SILVER", "GOLD"]:
                    cfg = SCHEDULER_HOURS["MCX"]
                elif sym_upper in ["SENSEX", "BANKEX"]:
                    cfg = SCHEDULER_HOURS["BSE"]
                else: continue
                
                # Strictly check market hours for background scheduler
                start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
                end_t   = datetime.strptime(cfg["end"],   "%H:%M:%S").time()
                if _secs(start_t) <= _secs(now.time()) <= _secs(end_t):
                    any_in_hours = True
                    break
        
        if not any_in_hours:
            # Market is closed.
            from core.utils import get_next_trading_day
            next_open = get_next_trading_day(now)
            sleep_sec = (next_open - now).total_seconds()
            
            print(f"[Scheduler] Market is closed. Hibernating till {next_open.strftime('%Y-%m-%d %H:%M')}...")
            _wake_event.wait(timeout=sleep_sec)
            _wake_event.clear()
        else:
            _wake_event.wait(timeout=20)
            _wake_event.clear()

def start_schedulers():
    t = threading.Thread(target=_main_scheduler_loop, daemon=True, name="MasterScheduler")
    t.start()
