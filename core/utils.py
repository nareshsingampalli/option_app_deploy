
import os
import time
import requests
import gzip
import io
import pandas as pd
import time
from core.config import CACHE_DIR
from upstox_client.rest import ApiException
import functools

def retry_api_call(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator to retry API calls on 429 (Rate Limit) and transient 5xx errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except ApiException as e:
                    # Retry on Rate Limit (429) or Server Errors (500, 502, 503, 504)
                    if e.status in (429, 500, 502, 503, 504):
                        retries += 1
                        if retries >= max_retries:
                            print(f"[Retry] Max retries reached. Last error: {e}")
                            raise e
                        print(f"[Retry] Error {e.status} detected. Retrying in {delay}s... ({retries}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise e
                except Exception as e:
                    # Also retry on generic transient connection errors
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    print(f"[Retry] Transient error {type(e).__name__}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
            return func(*args, **kwargs)
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

def get_last_trading_day(dt=None):
    """
    If dt (default today) is a weekend, returns the preceding Friday.
    Returns datetime object.
    """
    from datetime import timedelta
    if dt is None:
        dt = ist_now()
    
    # weekday(): 0=Monday, 5=Saturday, 6=Sunday
    wd = dt.weekday()
    if wd == 5: # Saturday
        return dt - timedelta(days=1)
    elif wd == 6: # Sunday
        return dt - timedelta(days=2)
    return dt

