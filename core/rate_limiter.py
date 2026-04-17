"""
Thread-safe token-bucket rate-limiter decorator.

Usage
-----
    from core.rate_limiter import rate_limited

    @rate_limited(max_calls=10, period=1.0)
    def my_api_call(...): ...
"""

import time
import threading
import functools
from typing import Callable


class _RateLimiter:
    """Internal token-bucket implementation."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period    = period
        self._lock     = threading.Lock()
        self._calls: list[float] = []

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sleep_for = 0
            with self._lock:
                now = time.monotonic()
                # Drop timestamps older than one period
                self._calls = [t for t in self._calls if now - t < self.period]
                
                if len(self._calls) >= self.max_calls:
                    # Calculate how long we must wait until the oldest slot expires
                    sleep_for = self.period - (now - self._calls[0])
                    # Mark the 'future' timestamp we will use after sleeping
                    call_ts = now + max(0, sleep_for)
                    self._calls = self._calls[1:]
                else:
                    call_ts = now
                
                self._calls.append(call_ts)
            
            if sleep_for > 0:
                time.sleep(sleep_for)
                
            return func(*args, **kwargs)
        return wrapper


def rate_limited(max_calls: int, period: float) -> Callable:
    """Decorator factory — wraps a function with rate-limiting."""
    return _RateLimiter(max_calls, period)
