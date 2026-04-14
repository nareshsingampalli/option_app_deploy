"""IntradayCandleFetcher — today's live candle data."""

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
            try:
                actual_interval = 1 if getattr(self, "_is_fallback", False) else self.interval
                from fetchers.historical import HistoricalCandleFetcher
                hist = HistoricalCandleFetcher()
                hist.interval = actual_interval
                
                from core.utils import get_last_trading_day
                from datetime import datetime, timedelta
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                prev_date = get_last_trading_day(dt - timedelta(days=1)).strftime("%Y-%m-%d")
                
                prev_df = hist.fetch_single(instrument_key, "minutes", actual_interval, prev_date, prev_date)
                
                if prev_df is not None and not prev_df.empty:
                    prev_row = prev_df.tail(1).copy()
                    
                    # ── Live-Mode OI Enhancement ────────────────────────────────
                    if prev_row["open_interest"].sum() == 0:
                        try:
                            print(f"[Intraday] Probing Market Quote for baseline OI: {instrument_key}")
                            quote_resp = self._quote_api.get_full_market_quote(instrument_key)
                            if quote_resp and quote_resp.data:
                                data_val = quote_resp.data.get(instrument_key)
                                if data_val and hasattr(data_val, 'oi'):
                                    prev_row["open_interest"] = float(data_val.oi)
                                    print(f"[Intraday] Found baseline OI: {data_val.oi}")
                        except Exception as q_err:
                            print(f"[Intraday] Market Quote probe failed: {q_err}")

                    # Save to persistent cache
                    row_to_save = prev_row.reset_index().to_dict('records')[0]
                    # Convert timestamp back to string for JSON
                    row_to_save['date'] = str(row_to_save['date'])
                    self._save_cache(cache_key, row_to_save)

                    df = pd.concat([prev_row, df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    print(f"[IntradayFetcher] Anchored {instrument_key} to baseline from {prev_date}")
            except Exception as e:
                print(f"[IntradayFetcher] Warning: Baseline anchoring failed for {instrument_key}: {e}")

        return df

    def warmup_cache(self, instruments: list, date_str: str):
        """Pre-fetch and cache baselines for a list of instruments in parallel (IO optimization)."""
        import concurrent.futures
        # If passed objects, extract keys. If passed strings (legacy/direct), use as is.
        keys = [inst.key if hasattr(inst, 'key') else inst for inst in instruments]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for key in keys:
                 executor.submit(self.get_candles, key, date_str)
        print(f"[IntradayFetcher] Cache warmup complete for {len(keys)} instruments.")

    def get_spot_candles(self, spot_key: str, date_str: str) -> pd.DataFrame | None:
        """Fetch spot/index candles for today. Empty result means market is closed/holiday."""
        df = self._fetch(spot_key, "minutes", self.interval, date_str=date_str)
        if df is None or df.empty:
            df = self._fetch(spot_key, "minutes", 1, date_str=date_str)
        return df
