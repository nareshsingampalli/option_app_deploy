import threading
import pandas as pd
from upstox_client.rest import ApiException

import core.config
from core.utils import ist_now
from core.rate_limiter import rate_limited
from core.config import UPSTOX_RATE_LIMIT_CALLS, UPSTOX_RATE_LIMIT_PERIOD
from fetchers.base import BaseCandleFetcher


class IntradayCandleFetcher(BaseCandleFetcher):
    """Fetches real-time 5-min candles for the current trading day."""

    def __init__(self, interval: int = core.config.CANDLE_INTERVAL_MINUTES):
        super().__init__(interval)
        import os
        self._cache_file = os.path.join(core.config.CACHE_DIR, "baseline_cache.json")
        self._baseline_cache = self._load_cache()
        self._data_cache = {}  # Store today's candles: {(key, interval): (timestamp, df)}
        import upstox_client
        self._quote_api = upstox_client.MarketQuoteApi(self._api_client)
        self._hist_fetcher = None # Lazy init
        self._warmup_lock = threading.Lock()
        self._fetching_keys = set()

    def _load_cache(self):
        import json, os
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_cache(self, key, row_dict):
        import json, os
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
        # Avoid growing into infinity: only keep today's entries
        today = ist_now().strftime("%Y-%m-%d")
        self._baseline_cache = {k: v for k, v in self._baseline_cache.items() if today in k}
        self._baseline_cache[key] = row_dict
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._baseline_cache, f)
        except: pass

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def _fetch(self, instrument_key: str, unit: str, interval: int, date_str: str | None = None) -> pd.DataFrame | None:
        import time
        now = time.time()
        cache_key = (instrument_key, interval)
        if cache_key in self._data_cache:
            ts, cached_df = self._data_cache[cache_key]
            if now - ts < 30:  # 30-second cache
                return cached_df.copy()

        from core.utils import retry_api_call
        
        @retry_api_call(max_retries=3)
        def _get():
            # In Upstox V2 SDK, get_intra_day_candle_data expects:
            # interval = '1minute' or '30minute'
            # api_version = '2.0'
            interval_str = f"{interval}minute" if unit == "minutes" else f"{interval}{unit}"
            return self._history_api.get_intra_day_candle_data(
                instrument_key, unit, interval
            )

        try:
            resp = _get()
            self.last_status = 200
            self.last_error_code = None
            df = self._process_response(resp, date_str=date_str)
            if df is not None and not df.empty:
                self._data_cache[cache_key] = (now, df)
            return df
        except ApiException as e:
            self.last_status = getattr(e, "status", None)
            try:
                import json
                body = json.loads(e.body)
                self.last_error_code = body.get("errors", [{}])[0].get("errorCode")
            except:
                self.last_error_code = None
            print(f"[IntradayFetcher] ApiException {instrument_key} (Status {self.last_status}, Code {self.last_error_code}): {e}")
            return None

    def get_candles(self, instrument_key: str, date_str: str, expiry_dt=None) -> pd.DataFrame | None:
        self._is_fallback = False
        df = self._fetch(instrument_key, "minutes", self.interval, date_str=date_str)
        if df is None or df.empty:
            print(f"[IntradayFetcher] {self.interval}-min unavailable, falling back to 1-min")
            df = self._fetch(instrument_key, "minutes", 1)
            self._is_fallback = True
        
        # Use persistent baseline cache if available
        cache_key = f"baseline_{instrument_key}_{date_str}"
        if cache_key in self._baseline_cache:
            prev_row_dict = self._baseline_cache[cache_key]
            if prev_row_dict:
                 prev_row = pd.DataFrame([prev_row_dict])
                 prev_row['date'] = pd.to_datetime(prev_row['timestamp'])
                 prev_row.set_index('date', inplace=True)
                 df = pd.concat([prev_row, df])
                 df = df[~df.index.duplicated(keep="last")].sort_index()
                 return df

        if df is not None and not df.empty:
            # If we reached here, it means we don't have a cached baseline and need a slow fetch.
            # Usually handled by warmup_cache in bulk.
            try:
                actual_interval = 1 if getattr(self, "_is_fallback", False) else self.interval
                if self._hist_fetcher is None:
                    from fetchers.historical import HistoricalCandleFetcher
                    self._hist_fetcher = HistoricalCandleFetcher()
                
                self._hist_fetcher.interval = actual_interval
                
                from core.utils import get_last_trading_day
                from datetime import datetime, timedelta
                
                # Rule 2: Deep Baseline Rollback (up to 5 days)
                # If 'yesterday' is empty (maintenance), keep looking back until we find a session.
                cursor_dt = datetime.strptime(date_str, "%Y-%m-%d")
                prev_df = None
                for _ in range(5):
                    cursor_dt = get_last_trading_day(cursor_dt - timedelta(days=1))
                    prev_date = cursor_dt.strftime("%Y-%m-%d")
                    prev_df = self._hist_fetcher.fetch_single(instrument_key, "minutes", actual_interval, prev_date, prev_date)
                    if prev_df is not None and not prev_df.empty:
                        break
                
                if prev_df is not None and not prev_df.empty:
                    prev_row = prev_df.tail(1).copy()
                    
                    # Single probe if not part of a batch warmup
                    if prev_row["open_interest"].sum() == 0:
                        try:
                            q_resp = self._quote_api.get_full_market_quote(instrument_key, api_version='2.0')
                            if q_resp and q_resp.data and instrument_key in q_resp.data:
                                val = q_resp.data[instrument_key]
                                prev_row["open_interest"] = float(val.oi)
                        except: pass

                    # Save to persistent cache
                    row_to_save = prev_row.reset_index().to_dict('records')[0]
                    row_to_save['date'] = str(row_to_save['date'])
                    self._save_cache(cache_key, row_to_save)

                    df = pd.concat([prev_row, df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
            except Exception as e:
                print(f"[IntradayFetcher] Warning: Deep Baseline anchoring failed for {instrument_key}: {e}")


        return df

    def warmup_cache(self, instruments: list, date_str: str):
        """Pre-fetch and cache baselines in bulk (IO optimization)."""
        import concurrent.futures
        keys = [inst.key if hasattr(inst, 'key') else inst for inst in instruments]
        
        # 1. Identify missing baselines
        missing_keys = [k for k in keys if f"baseline_{k}_{date_str}" not in self._baseline_cache]
        if not missing_keys:
            return

        print(f"[IntradayFetcher] Warming up {len(missing_keys)} missing baselines...")
        
        def do_warmup(key):
            with self._warmup_lock:
                if key in self._fetching_keys: return
                self._fetching_keys.add(key)
            try:
                self.get_candles(key, date_str)
            finally:
                with self._warmup_lock: self._fetching_keys.remove(key)

        # 2. Parallel fetch (covers get_candles -> fetch_single + self-caching)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lambda k: do_warmup(k), missing_keys)

        # 3. Batch Quote Probe for remaining 0-OI baselines
        # Identify those that still have 0 OI after the fetch
        needs_oi_probe = []
        for key in missing_keys:
            cache_key = f"baseline_{key}_{date_str}"
            row = self._baseline_cache.get(cache_key)
            if row and row.get("open_interest", 0) == 0:
                needs_oi_probe.append(key)
        
        if needs_oi_probe:
            print(f"[IntradayFetcher] Batch probing OI for {len(needs_oi_probe)} instruments...")
            # Upstox allows up to 50 instruments per quote call
            for i in range(0, len(needs_oi_probe), 50):
                batch = needs_oi_probe[i:i+50]
                try:
                    q_resp = self._quote_api.get_full_market_quote(",".join(batch), api_version='2.0')

                    if q_resp and q_resp.data:
                        for k, val in q_resp.data.items():
                            c_key = f"baseline_{k}_{date_str}"
                            if c_key in self._baseline_cache:
                                self._baseline_cache[c_key]["open_interest"] = float(val.oi)
                                # Silent update - we don't need to re-save the whole file every time
                except Exception as e:
                    print(f"[IntradayFetcher] Batch OI probe error: {e}")
            
            # Final flush of the updated cache
            try:
                import json
                with open(self._cache_file, 'w') as f:
                    json.dump(self._baseline_cache, f)
            except: pass

        print(f"[IntradayFetcher] Cache warmup complete.")

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        """Fetch spot/index candles for today with historical baseline."""
        df = self._fetch(spot_key, "minutes", self.interval, date_str=date_str)
        if df is None or df.empty:
            df = self._fetch(spot_key, "minutes", 1, date_str=date_str)
        
        if df is None or df.empty:
            return None

        # Prepend baseline for ROC/IV anchoring
        cache_key = f"baseline_{spot_key}_{date_str}"
        if cache_key in self._baseline_cache:
            prev_row_dict = self._baseline_cache[cache_key]
            if prev_row_dict:
                 prev_row = pd.DataFrame([prev_row_dict])
                 prev_row['date'] = pd.to_datetime(prev_row['timestamp'])
                 prev_row.set_index('date', inplace=True)
                 df = pd.concat([prev_row, df])
                 return df[~df.index.duplicated(keep="last")].sort_index()

        # If not in cache, fallback to slow fetch (instrument_key = spot_key)
        try:
            if self._hist_fetcher is None:
                from fetchers.historical import HistoricalCandleFetcher
                self._hist_fetcher = HistoricalCandleFetcher()
            
            from core.utils import get_last_trading_day
            from datetime import datetime, timedelta
            
            # Rule 2: Deep Baseline Rollback for Spot
            cursor_dt = datetime.strptime(date_str, "%Y-%m-%d")
            prev_df = None
            for _ in range(5):
                cursor_dt = get_last_trading_day(cursor_dt - timedelta(days=1))
                prev_date = cursor_dt.strftime("%Y-%m-%d")
                prev_df = self._hist_fetcher.fetch_single(spot_key, "minutes", self.interval, prev_date, prev_date)
                if prev_df is not None and not prev_df.empty:
                    break

            if prev_df is not None and not prev_df.empty:
                prev_row = prev_df.tail(1).copy()
                # Save to cache
                row_to_save = prev_row.reset_index().to_dict('records')[0]
                row_to_save['date'] = str(row_to_save['date'])
                self._save_cache(cache_key, row_to_save)
                
                df = pd.concat([prev_row, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
        except: pass

        return df


