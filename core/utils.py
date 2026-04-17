
import os
import time
import requests
import gzip
import io
import pandas as pd
from core.config import CACHE_DIR
from upstox_client.rest import ApiException
import functools
import threading

_api_rl_lock = threading.Lock()
_api_rl_wait_until = 0.0

def retry_api_call(max_retries: int = 3, initial_delay: float = 2.0, max_duration: float = 600.0):
    """Decorator to retry API calls on 429 (Rate Limit) and transient 5xx errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _api_rl_wait_until
            start_time = time.time()
            retries = 0
            delay = initial_delay
            
            while True:
                # 1. Check if we've reached the 5-min limit (280s to finish before next 300s cycle)
                if time.time() - start_time >= max_duration:
                    print(f"[Retry] Request stale ({max_duration}s elapsed). Dropping to make way for new scheduler requests.")
                    raise TimeoutError(f"API request timed out due to max_duration ({max_duration}s).")

                # 2. Global Rate Limit check - pause if another thread requested a wait
                with _api_rl_lock:
                    wait_time = _api_rl_wait_until - time.time()
                
                if wait_time > 0:
                    time.sleep(min(wait_time, max_duration - (time.time() - start_time)))
                    continue # Re-evaluate duration and lock after sleeping

                # 3. Call the API
                try:
                    return func(*args, **kwargs)
                except ApiException as e:
                    # Retry on Rate Limit (429) or Server Errors (500, 502, 503, 504)
                    if getattr(e, "status", None) in (429, 500, 502, 503, 504):
                        retries += 1
                        if retries > max_retries:
                            print(f"[Retry] Maximum retry attempts reached ({max_retries}). Still hitting 429. Continuing backoff...")
                        
                        print(f"[Retry] API Error {e.status} detected. All threads pausing for {delay}s... (Attempt {retries}/{max_retries})")
                        
                        with _api_rl_lock:
                            new_wait_until = time.time() + delay
                            if new_wait_until > _api_rl_wait_until:
                                _api_rl_wait_until = new_wait_until
                        
                        # Immediate sleep for this thread to synchronize with others
                        time.sleep(delay)
                        
                        delay = min(delay * 2, 60.0)  # Exponential backoff capped at 60s
                    else:
                        raise e
                except Exception as e:
                    # Also retry on generic transient connection errors
                    retries += 1
                    if retries > max_retries:
                        print(f"[Retry] Resetting delay to {initial_delay}s after {max_retries} attempts.")
                        retries = 1
                        delay = initial_delay
                    
                    print(f"[Retry] Transient error {type(e).__name__} during API call. Retrying in {delay}s...")
                    sleep_time = min(delay, max_duration - (time.time() - start_time))
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    delay *= 2
                    
        return wrapper
    return decorator

def get_instrument_df(url: str, exchange: str) -> pd.DataFrame:
    """
    Downloads and caches instrument files for one day.
    Supports .csv.gz and .json.gz
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    filename = f"{exchange.lower()}_instruments"
    if url.endswith(".csv.gz"):
        filename += ".csv.gz"
    else:
        filename += ".json.gz"
        
    cache_path = os.path.join(CACHE_DIR, filename)

    # Check if cache exists and is less than 24 hours old
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) < 86400: # 24 hours
            print(f"[Cache] Using cached {exchange} instruments.")
            if cache_path.endswith(".csv.gz"):
                with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                    return pd.read_csv(f)
            else:
                with gzip.open(cache_path, "rb") as f:
                    return pd.read_json(f)

    # Download if not cached or expired
    print(f"[Cache] Downloading {exchange} instruments from Upstox...")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    
    with open(cache_path, "wb") as f:
        f.write(r.content)

    if url.endswith(".csv.gz"):
        content = gzip.decompress(r.content).decode("utf-8")
        return pd.read_csv(io.StringIO(content))
    else:
        with gzip.open(io.BytesIO(r.content), "rb") as f:
            return pd.read_json(f)

def ist_now():
    """Returns current time in IST (required for cross-platform/cloud servers)."""
    import pytz
    from datetime import datetime
    return datetime.now(pytz.timezone('Asia/Kolkata'))

# ── Holiday Handling ─────────────────────────────────────────────────────────
# Holidays are NOT hardcoded. They are discovered at runtime by probing 
# the Upstox API — empty response = holiday = auto roll-over to previous day.
# ── Trading Day Arithmetic ───────────────────────────────────────────────────
# No assumptions are made about holidays or weekends. The system attempts 
# to fetch data for any requested date. If the response is empty, the 
# high-level retry logic (in routes.py) handles the step-back.

def get_last_trading_day(dt=None):
    """Returns the calendar day immediately preceding dt."""
    from datetime import timedelta
    if dt is None:
        dt = ist_now()
    return dt - timedelta(days=1)

def get_prev_trading_day(date_str: str = None) -> str:
    """
    Returns the previous weekday (Mon–Fri) before the given date string (YYYY-MM-DD).
    Skips Saturday and Sunday automatically.
    Exchange holidays are NOT hardcoded — they are discovered at runtime via
    empty API responses, which trigger another rollback automatically.
    """
    from datetime import datetime, timedelta
    if date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        dt = ist_now().replace(tzinfo=None)
    dt -= timedelta(days=1)
    while dt.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
        dt -= timedelta(days=1)
    return dt.strftime("%Y-%m-%d")

def get_next_trading_day(dt=None):
    """Returns the calendar day immediately following dt."""
    from datetime import timedelta
    if dt is None:
        dt = ist_now()
    return (dt + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)


