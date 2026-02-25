import os
import signal
import pandas as pd
from datetime import datetime, timedelta
import upstox_client
from upstox_client.rest import ApiException
import time
import requests
import urllib.parse

def timeout_handler(signum, frame):
    raise TimeoutError("API request timed out")

class BaseCandleFetcher:
    """Base class for candle fetching shared across intraday and historical."""
    def __init__(self, access_token=None):
        self.access_token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not self.access_token:
            # Fallback for sessions where .env might not be loaded yet or for quick tests
            self.access_token = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI4Q0FRNzUiLCJqdGkiOiI2OTllNGU2MmIwNWNhMTYwMDE3ZGUzYTIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzcxOTgyNDM0LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzIwNTY4MDB9.BiPUE1cVru3fLGEJKItJiL5SFBV3kgqINmveiLddsDc'
        
        if not self.access_token or self.access_token == 'None':
            raise ValueError("UPSTOX_ACCESS_TOKEN not found in environment or arguments.")
        
        self.configuration = upstox_client.Configuration()
        self.configuration.access_token = self.access_token
        self.api_client = upstox_client.ApiClient(self.configuration)
        self.history_api = upstox_client.HistoryV3Api(self.api_client)

    def _process_response(self, response):
        """Standard processing of Upstox API response into a DataFrame."""
        try:
            candles = None
            data = getattr(response, 'data', None)
            if data is not None:
                candles = getattr(data, 'candles', None)
            
            # Fallback for dict response
            if candles is None and isinstance(response, dict):
                data = response.get('data', None)
                if isinstance(data, dict) and 'candles' in data:
                    candles = data['candles']

            if not candles:
                return None

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
            df['date'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('date').sort_index()
            
            for col in ["open", "high", "low", "close", "volume", "open_interest"]:
                df[col] = pd.to_numeric(df[col])
            
            return df
        except Exception as e:
            print(f"[ERROR] Error processing API response: {e}")
            return None

    def _execute_with_retry(self, func, *args, **kwargs):
        """Executes an API function with retries on rate limiting (429)."""
        max_retries = 5
        import random
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ApiException as e:
                if e.status == 429:
                    # Exponential backoff with jitter
                    wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                    print(f"[RateLimit] Hit 429. Waiting {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                raise e
            except Exception as e:
                # For requests-based calls (ExpiredCandleFetcher)
                if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429:
                     wait_time = (attempt + 1) * 2
                     time.sleep(wait_time)
                     continue
                raise e
        return None

class IntradayCandleFetcher(BaseCandleFetcher):
    """Fetches candle data for the current trading day."""
    def fetch(self, instrument_key, timeframe="minutes", interval_num=1):
        """
        Uses get_intra_day_candle_data for the present day.
        """
        unit = str(timeframe).lower()
        if unit == 'hour': unit = 'hours'
        if unit == 'minute': unit = 'minutes'
        if unit == 'hours':
            unit = 'minutes'
            interval_num = int(interval_num) * 60
        if not unit.endswith('s'):
            unit += 's'
        
        try:
            response = self._execute_with_retry(
                self.history_api.get_intra_day_candle_data,
                instrument_key, unit, str(interval_num)
            )
            return self._process_response(response)
        except Exception as e:
            print(f"[ERROR] Intraday fetch failed: {e}")
            return None

class HistoricalCandleFetcher(BaseCandleFetcher):
    """Fetches candle data for historical periods with chunking support."""
    
    def _fetch_single(self, instrument_key, unit, interval_num, to_date, from_date):
        """Internal method to fetch a single chunk of data."""
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(90)
                
            response = self._execute_with_retry(
                self.history_api.get_historical_candle_data1,
                instrument_key, unit, str(interval_num), to_date, from_date
            )
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
            return self._process_response(response)
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            print(f"[ERROR] Historical fetch failed: {e}")
            return None

    def fetch(self, instrument_key, timeframe="days", interval_num=1, lookback_days=90):
        """
        Uses get_historical_candle_data1 for past data.
        timeframe: 'minutes', 'hours', 'days', 'weeks', 'months'
        lookback_days: number of days to go back from today.
        
        For minutes/hours: Automatically chunks into 30-day windows.
        For days/weeks/months: Single request.
        """
        unit = str(timeframe).lower()
        if unit == 'hour': unit = 'hours'
        if unit == 'minute': unit = 'minutes'
        if not unit.endswith('s'):
            unit += 's'
        
        # For intraday units (minutes/hours), chunk into 30-day windows
        if unit in ['minutes', 'hours'] and lookback_days > 30:
            return self._fetch_chunked(instrument_key, unit, interval_num, lookback_days)
        
        # For daily/weekly/monthly, no chunking needed
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        return self._fetch_single(instrument_key, unit, interval_num, to_date, from_date)

    def _fetch_chunked(self, instrument_key, unit, interval_num, lookback_days, chunk_size=30):
        """
        Fetches data in chunks for minute/hour data to avoid API date range limits.
        Returns concatenated DataFrame.
        """
        all_dfs = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        current_end = end_date
        while current_end > start_date:
            current_start = max(current_end - timedelta(days=chunk_size), start_date)
            
            to_date_str = current_end.strftime('%Y-%m-%d')
            from_date_str = current_start.strftime('%Y-%m-%d')
            
            print(f"[INFO] Fetching chunk: {from_date_str} to {to_date_str}")
            df = self._fetch_single(instrument_key, unit, interval_num, to_date_str, from_date_str)
            
            if df is not None and len(df) > 0:
                all_dfs.append(df)
            
            current_end = current_start - timedelta(days=1)
            time.sleep(0.5)  # Rate limit safety
        
        if not all_dfs:
            return None
        
        # Concatenate and remove duplicates
        combined = pd.concat(all_dfs)
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined.sort_index()

    def fetch_3months(self, instrument_key, timeframe, interval_num):
        """
        Convenience method to fetch exactly 3 months (90 days) of data.
        """
        return self.fetch(instrument_key, timeframe, interval_num, lookback_days=90)

class ExpiredCandleFetcher(BaseCandleFetcher):
    """Fetches candle data for expired instruments using direct HTTP REST API calls."""
    def __init__(self, access_token=None):
        super().__init__(access_token)
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

    def fetch_contracts(self, underlying_key, expiry_date):
        """Fetch expired option contracts for a given underlying and expiry date."""
        try:
            safe_key = urllib.parse.quote(underlying_key)
            url = f"https://api.upstox.com/v2/expired-instruments/option/contract?instrument_key={safe_key}&expiry_date={expiry_date}"
            
            response = self._execute_with_retried_get(url)
            if response and response.status_code == 200:
                return response.json().get('data', [])
            return []
        except Exception as e:
            print(f"[ERROR] fetch_contracts Exception: {e}")
            return []

    def _execute_with_retried_get(self, url):
        """Helper for requests.get with 429 retries."""
        max_retries = 5
        import random
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=self.headers, timeout=15)
                if resp.status_code == 429:
                    wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                    print(f"[RateLimit] Hit 429 (REST). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                return resp
            except Exception as e:
                if attempt == max_retries - 1: raise e
                time.sleep(2)
        return None

    def fetch_expiries(self, underlying_key):
        """Fetch available expiries for an underlying index/instrument."""
        try:
            safe_key = urllib.parse.quote(underlying_key)
            url = f"https://api.upstox.com/v2/expired-instruments/expiries?instrument_key={safe_key}"
            
            response = self._execute_with_retried_get(url)
            if response and response.status_code == 200:
                return response.json().get('data', [])
            return []
        except Exception as e:
            print(f"[ERROR] fetch_expiries Exception: {e}")
            return []

    def fetch_candle_data(self, instrument_key, interval_str, to_date, from_date):
        """Fetch historical candle data for an expired instrument."""
        try:
            safe_key = urllib.parse.quote(instrument_key)
            url = f"https://api.upstox.com/v2/expired-instruments/historical-candle/{safe_key}/{interval_str}/{to_date}/{from_date}"
            
            response = self._execute_with_retried_get(url)
            if response and response.status_code == 200:
                return self._process_response(response.json())
            return None
        except Exception as e:
            print(f"[ERROR] fetch_candle_data Exception: {e}")
            return None
