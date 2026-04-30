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
        self._cache_file_user = os.path.join(core.config.CACHE_DIR, "baseline.json")
        self._cache_file_data = os.path.join(core.config.CACHE_DIR, "internal_data.json")
        self._baseline_user = self._load_json(self._cache_file_user) # Nested: {date: {symbol: {type: {key: oi}}}}
        self._baseline_data = self._load_json(self._cache_file_data) # Flat: {f"{key}_{date}": candles}
        self._data_cache = {}  
        import upstox_client
        self._quote_api = upstox_client.MarketQuoteApi(self._api_client)
        self._hist_fetcher = None 
        self._warmup_lock = threading.Lock()
        self._fetching_keys = set()

    def _load_json(self, path):
        import json, os
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_cache_entry(self, inst_obj, date_str, candle_list):
        """Saves current OI to baseline.json and full list to internal_data.json."""
        import json, os
        if not candle_list: return
        
        last_oi = candle_list[-1].get("open_interest", 0.0)
        
        # User structure: SYMBOL -> TYPE -> KEY: OI
        if hasattr(inst_obj, 'symbol') and hasattr(inst_obj, 'option_type'):
            base_name = inst_obj.symbol.split(' ')[0]
            opt_type = inst_obj.option_type
            key = inst_obj.key
        else:
            base_name = str(inst_obj).split('|')[-1]
            opt_type = "INDEX"
            key = str(inst_obj)

        if date_str not in self._baseline_user: self._baseline_user[date_str] = {}
        if base_name not in self._baseline_user[date_str]: 
            self._baseline_user[date_str][base_name] = {"CE": {}, "PE": {}, "INDEX": {}}
        
        self._baseline_user[date_str][base_name][opt_type][key] = last_oi
        self._baseline_data[f"{key}_{date_str}"] = candle_list
        
        try:
            os.makedirs(os.path.dirname(self._cache_file_user), exist_ok=True)
            with open(self._cache_file_user, 'w') as f:
                json.dump(self._baseline_user, f, indent=2)
            with open(self._cache_file_data, 'w') as f:
                json.dump(self._baseline_data, f) # Compact internal data
        except: pass

    @rate_limited(max_calls=UPSTOX_RATE_LIMIT_CALLS, period=UPSTOX_RATE_LIMIT_PERIOD)
    def _fetch(self, instrument_key: str, unit: str, interval: int, date_str: str | None = None) -> pd.DataFrame | None:
        self._sync_token()
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
        
        # Rule 1: Use internal data cache for calculations
        cache_key = f"{instrument_key}_{date_str}"
        if cache_key in self._baseline_data:
            prev_data = self._baseline_data[cache_key]
            if prev_data:
                 prev_df = pd.DataFrame(prev_data)
                 prev_df['date'] = pd.to_datetime(prev_df['timestamp'])
                 prev_df.set_index('date', inplace=True)
                 df = pd.concat([prev_df, df])
                 df = df[~df.index.duplicated(keep="last")].sort_index()
                 return df

        if df is not None and not df.empty:
            # Rule 2: Fetch and persist if not found
            try:
                actual_interval = 1 if getattr(self, "_is_fallback", False) else self.interval
                if self._hist_fetcher is None:
                    from fetchers.historical import HistoricalCandleFetcher
                    self._hist_fetcher = HistoricalCandleFetcher()
                
                self._hist_fetcher.interval = actual_interval
                from core.utils import get_last_trading_day
                from datetime import datetime, timedelta
                
                cursor_dt = datetime.strptime(date_str, "%Y-%m-%d")
                prev_df = None
                for _ in range(30):
                    cursor_dt = get_last_trading_day(cursor_dt - timedelta(days=1))
                    prev_date = cursor_dt.strftime("%Y-%m-%d")
                    prev_df = self._hist_fetcher.fetch_single(instrument_key, "minutes", actual_interval, prev_date, prev_date)
                    if prev_df is not None and not prev_df.empty:
                        break
                
                if prev_df is not None and not prev_df.empty:
                    rows_to_save = prev_df.reset_index().to_dict('records')
                    for r in rows_to_save: r['date'] = str(r['date'])
                    
                    # Store in nested structure via helper
                    # We pass the key as fallback in case we don't have the instrument object here
                    self._save_cache_entry(instrument_key, date_str, rows_to_save)
                    
                    df = pd.concat([prev_df, df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
            except Exception as e:
                print(f"[IntradayFetcher] Warning: Baseline fetch failed for {instrument_key}: {e}")



        return df

    def warmup_cache(self, instruments: list, date_str: str):
        """Pre-fetch and cache baselines in bulk (IO optimization)."""
        import concurrent.futures
        keys = [inst.key if hasattr(inst, 'key') else inst for inst in instruments]
        
        # 1. Identify missing baselines
        missing_keys = [k for k in keys if f"{k}_{date_str}" not in self._baseline_data]
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
        needs_oi_probe = []
        for key in missing_keys:
            flat_key = f"{key}_{date_str}"
            data = self._baseline_data.get(flat_key)
            if data and isinstance(data, list):
                if data[-1].get("open_interest", 0) == 0:
                    needs_oi_probe.append(key)
        
        if needs_oi_probe:
            print(f"[IntradayFetcher] Batch probing OI for {len(needs_oi_probe)} instruments...")
            for i in range(0, len(needs_oi_probe), 50):
                batch = needs_oi_probe[i:i+50]
                try:
                    q_resp = self._quote_api.get_full_market_quote(",".join(batch), api_version='2.0')
                    if q_resp and q_resp.data:
                        for k, val in q_resp.data.items():
                            flat_key = f"{k}_{date_str}"
                            if flat_key in self._baseline_data:
                                # Update internal data list
                                self._baseline_data[flat_key][-1]["open_interest"] = float(val.oi)
                                # Update user-facing nested summary (we need to search for it)
                                # For simplicity, we'll just flush the whole cache at the end
                except Exception as e:
                    print(f"[IntradayFetcher] Batch OI probe error: {e}")
            
            # Re-sync user summary from patched data
            self._resync_user_summary(instruments, date_str)
            
            # Flush both to disk
            try:
                import json
                with open(self._cache_file_user, 'w') as f:
                    json.dump(self._baseline_user, f, indent=2)
                with open(self._cache_file_data, 'w') as f:
                    json.dump(self._baseline_data, f)
            except: pass

        self._resync_user_summary(instruments, date_str)
        print(f"[IntradayFetcher] Cache warmup complete.")

    def _resync_user_summary(self, instruments, date_str):
        """Helper to sync the user-facing baseline.json after a batch OI probe."""
        for inst in instruments:
            if not hasattr(inst, 'key'): continue
            flat_key = f"{inst.key}_{date_str}"
            data = self._baseline_data.get(flat_key)
            if data:
                last_oi = data[-1].get("open_interest", 0.0)
                base_name = inst.symbol.split(' ')[0] if hasattr(inst, 'symbol') else str(inst).split('|')[-1]
                opt_type  = inst.option_type if hasattr(inst, 'option_type') else "INDEX"
                
                if date_str not in self._baseline_user: self._baseline_user[date_str] = {}
                if base_name not in self._baseline_user[date_str]: self._baseline_user[date_str][base_name] = {"CE": {}, "PE": {}, "INDEX": {}}
                self._baseline_user[date_str][base_name][opt_type][inst.key] = last_oi

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        """Fetch spot/index candles for today with historical baseline."""
        df = self._fetch(spot_key, "minutes", self.interval, date_str=date_str)
        if df is None or df.empty:
            df = self._fetch(spot_key, "minutes", 1, date_str=date_str)
        
        if df is None or df.empty:
            return None

        # Rule 1: Use internal data cache for Spot calculations
        cache_key = f"{spot_key}_{date_str}"
        if cache_key in self._baseline_data:
            prev_data = self._baseline_data[cache_key]
            if prev_data:
                prev_df = pd.DataFrame(prev_data)
                prev_df['date'] = pd.to_datetime(prev_df['timestamp'])
                prev_df.set_index('date', inplace=True)
                df = pd.concat([prev_df, df])
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
            for _ in range(30):
                cursor_dt = get_last_trading_day(cursor_dt - timedelta(days=1))
                prev_date = cursor_dt.strftime("%Y-%m-%d")
                prev_df = self._hist_fetcher.fetch_single(spot_key, "minutes", self.interval, prev_date, prev_date)
                if prev_df is not None and not prev_df.empty:
                    break

            if prev_df is not None and not prev_df.empty:
                rows_to_save = prev_df.reset_index().to_dict('records')
                for r in rows_to_save: r['date'] = str(r['date'])
                
                # Store in nested structure
                self._save_cache_entry(spot_key, date_str, rows_to_save)
                
                df = pd.concat([prev_df, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
        except: pass

        return df


