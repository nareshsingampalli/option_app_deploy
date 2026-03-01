"""Background market-hours schedulers (one thread per exchange)."""

from __future__ import annotations

import sys
import subprocess
import threading
import time
from datetime import datetime, timedelta

from dashboard import app, socketio
from core.config import SCHEDULER_HOURS
from core.utils import ist_now

# Per-symbol fetch locks — dynamic
_locks_lock = threading.Lock()
_fetch_locks: dict[str, threading.Lock] = {}

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

    while True:
        now = ist_now()
        cur_min = now.minute
        active_symbols = get_active_symbols()
        
        # Always ensure base symbols are "active" for background maintenance
        if "NIFTY" not in active_symbols: active_symbols.append("NIFTY")
        if "CRUDEOIL" not in active_symbols: active_symbols.append("CRUDEOIL")

        for symbol in active_symbols:
            # Determine exchange and config
            if symbol in NSE_INDEX_KEYS or symbol == "NIFTY":
                cfg = SCHEDULER_HOURS["NSE"]
                prefix = "NSE"
            elif symbol in MCX_FUT_KEYS or symbol == "CRUDEOIL":
                cfg = SCHEDULER_HOURS["MCX"]
                prefix = "MCX"
            else:
                continue # Unknown symbol

            # Check market hours
            start_t = datetime.strptime(cfg["start"], "%H:%M:%S").time()
            end_t   = datetime.strptime(cfg["end"],   "%H:%M:%S").time()
            
            if _secs(start_t) <= _secs(now.time()) <= _secs(end_t):
                # 5-minute cycle
                is_startup = (now.hour == 9 and 16 <= now.minute <= 20)
                is_regular = (now.minute % 5 == 1)
                
                if (is_startup or is_regular) and last_fetch_times.get(symbol) != cur_min:
                    threading.Thread(
                        target=_run_fetch, 
                        args=(symbol, prefix),
                        daemon=True
                    ).start()
                    last_fetch_times[symbol] = cur_min
        
        time.sleep(20)

def start_schedulers():
    t = threading.Thread(target=_main_scheduler_loop, daemon=True, name="MasterScheduler")
    t.start()
