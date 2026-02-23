import os
import re
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import requests
import io
import gzip
import time
import sys
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from candle_fetchers import HistoricalCandleFetcher, ExpiredCandleFetcher, IntradayCandleFetcher

# Load environment variables
load_dotenv()
load_dotenv("/home/ubuntu/refactor_app/.env")

def get_spot_key():
    """Get the instrument key for Crude Oil Future (March 26)."""
    return "MCX_FO|472789"

def black_scholes_call(S, K, T, r, sigma):
    try:
        if T <= 0: return max(0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except: return 0

def black_scholes_put(S, K, T, r, sigma):
    try:
        if T <= 0: return max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except: return 0

def implied_volatility(price, S, K, T, r, option_type):
    sigma = 0.5
    try:
        for i in range(20):
            if option_type == 'CE':
                P = black_scholes_call(S, K, T, r, sigma)
            else:
                P = black_scholes_put(S, K, T, r, sigma)
            
            diff = price - P
            if abs(diff) < 1e-4:
                return sigma
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if vega == 0: break
            sigma = sigma + diff / vega
            if sigma <= 0: sigma = 0.01
            if sigma > 5: sigma = 5
        return sigma
    except:
        return 0

def get_option_chain_instruments(spot_price, num_strikes=5, reference_date=None, symbol='CRUDEOIL'):
    """
    Fetch MCX option chain instruments for a given commodity symbol.
    Supports: CRUDEOIL, SILVER, NATURALGAS, GOLD (and any other MCX symbol).
    
    Filters trading_symbol by:  ^{SYMBOL} \\d+ (CE|PE) ...
    D-1 expiry rule: switches to next expiry one day before current expiry.
    
    NOTE: spot_price and candle analysis still use get_spot_key() — unchanged.
    """
    strike_step = 50
    atm = round(spot_price / strike_step) * strike_step

    # ATM +3 / -3 = 7 strikes total
    target_strikes = [atm + strike_step * i for i in range(-3, 4)]
    print(f"[{symbol}] Spot: {spot_price} | ATM: {atm} | Strikes: {target_strikes}")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = 'https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz'
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        with gzip.open(io.BytesIO(r.content), 'rb') as f:
            df = pd.read_json(f)

        import re as _re

        # ── Step 1: Broad filter to determine expiry month (D-1 logic) ──────────
        # Filter CE/PE options whose trading_symbol starts with the commodity name
        broad = df[
            df['instrument_type'].isin(['CE', 'PE']) &
            df['trading_symbol'].str.startswith(symbol + ' ')
        ].copy()

        if broad.empty:
            print(f"[{symbol}] No CE/PE options found in MCX instruments.")
            return [], None, False

        broad['expiry_dt'] = pd.to_datetime(broad['expiry'], unit='ms')

        ref_dt = pd.to_datetime(reference_date) if reference_date else datetime.now()

        # D-1 rule: on day before expiry, switch to next month
        effective_cutoff = pd.Timestamp(ref_dt) + pd.Timedelta(days=1)
        valid_expiries = (
            broad[broad['expiry_dt'] > effective_cutoff]['expiry_dt']
            .sort_values().unique()
        )
        if len(valid_expiries) == 0:
            valid_expiries = broad['expiry_dt'].sort_values().unique()
            if len(valid_expiries) == 0:
                return [], None, False

        target_expiry = valid_expiries[0]

        # Derive month and year from target expiry
        commodity = symbol                              # e.g. "CRUDEOIL", "NATURALGAS"
        month     = target_expiry.strftime("%b").upper()  # e.g. "MAR", "MAY"
        year      = target_expiry.strftime("%y")          # e.g. "26"

        print(f"[{commodity}] Expiry: {target_expiry.date()} | {month} {year} (cutoff: {effective_cutoff.date()})")

        # ── Step 2: Precise regex — same structure as user-provided pattern ──────
        # Matches FUT contracts AND CE/PE option contracts for the target month
        # e.g. "NATURALGAS FUT 25 MAR 26"  or  "NATURALGAS 330 CE 25 MAR 26"
        precise_pattern = rf"^{commodity} (FUT \d{{1,2}}|\d+ (CE|PE) \d{{1,2}}) {month} {year}$"

        opts_expiry = df[
            df['instrument_type'].isin(['CE', 'PE']) &
            df['trading_symbol'].str.contains(precise_pattern, regex=True)
        ].copy()
        opts_expiry['expiry_dt'] = pd.to_datetime(opts_expiry['expiry'], unit='ms')
        opts_expiry = opts_expiry[opts_expiry['expiry_dt'] == target_expiry].copy()
        opts_expiry['strike'] = opts_expiry['strike_price'].astype(float)

        if opts_expiry.empty:
            print(f"[{commodity}] No options matched for {month} {year}.")
            return [], None, False

        selected_instruments = []
        expiry_map = {}  # { underlying: expiry_str } e.g. { 'CRUDEOIL': '17 MAR 26' }

        for strike in target_strikes:
            for opt_type in ['CE', 'PE']:
                match = opts_expiry[
                    (opts_expiry['instrument_type'] == opt_type) &
                    (opts_expiry['strike'] == float(strike))
                ]
                if match.empty:
                    subset = opts_expiry[opts_expiry['instrument_type'] == opt_type].copy()
                    subset['dist'] = abs(subset['strike'] - strike)
                    match = subset.nsmallest(1, 'dist')
                    if not match.empty:
                        found_strike = match.iloc[0]['strike']
                        print(f"  Exact strike {strike}{opt_type} not found, using nearest: {found_strike}")
                    else:
                        print(f"  No {opt_type} found near strike {strike}")
                        continue

                row = match.iloc[0]
                sym = row['trading_symbol']

                # Parse expiry string from trading_symbol: "CRUDEOIL 6600 CE 17 MAR 26" → "17 MAR 26"
                m = re.search(r'(?:CE|PE)\s+(.+)$', sym)
                expiry_str = m.group(1).strip() if m else target_expiry.strftime('%d %b %y').upper()

                expiry_map[sym.split()[0]] = expiry_str  # e.g. 'CRUDEOIL' -> '17 MAR 26'

                selected_instruments.append({
                    'symbol':     sym,
                    'key':        row['instrument_key'],
                    'type':       opt_type,
                    'strike':     row['strike'],
                    'expiry':     target_expiry,
                    'expiry_str': expiry_str,   # human-readable e.g. "17 MAR 26"
                })

        print(f"Selected {len(selected_instruments)} instruments: {[i['symbol'] for i in selected_instruments]}")
        print(f"Expiry map: {expiry_map}")
        return selected_instruments, target_expiry, False

    except Exception as e:
        print(f"Error fetching MCX chain: {e}")
        return [], None, False


class MarketDataStrategy(ABC):
    @abstractmethod
    def get_spot_price(self, date_str, time_str): pass
    @abstractmethod
    def get_instruments(self, spot_price, date_str): pass
    @abstractmethod
    def get_candle_data(self, key, from_date, to_date): pass

class LiveStrategy(MarketDataStrategy):
    def __init__(self):
        self.fetcher = IntradayCandleFetcher()
        self.spot_key = get_spot_key()
    def get_spot_price(self, ds, ts):
        df = self.fetcher.fetch(self.spot_key, "minutes", 5)
        return df['close'].iloc[-1] if df is not None and not df.empty else None
    def get_instruments(self, sp, ds):
        return get_option_chain_instruments(sp, num_strikes=5, reference_date=ds)
    def get_candle_data(self, key, fd, td):
        return self.fetcher.fetch(key, "minutes", 5)

class HistoricalStrategy(MarketDataStrategy):
    def __init__(self):
        self.fetcher = HistoricalCandleFetcher()
        self.spot_key = get_spot_key()
    def get_spot_price(self, ds, ts):
        spot_price = None
        if ts:
             df = self.fetcher._fetch_single(self.spot_key, "minutes", 5, ds, ds)
             if df is not None and not df.empty:
                dt = datetime.strptime(f"{ds} {ts}", "%Y-%m-%d %H:%M")
                try: spot_price = df.iloc[df.index.get_indexer([dt], method='nearest')[0]]['close']
                except: pass
        if spot_price is None:
            df = self.fetcher.fetch(self.spot_key, timeframe="days", lookback_days=30)
            if df is not None and not df.empty:
                if ds in df.index.strftime('%Y-%m-%d'): spot_price = df.loc[ds]['close']
                else: spot_price = df['close'].iloc[-1]
        return spot_price
    def get_instruments(self, sp, ds):
        return get_option_chain_instruments(sp, num_strikes=5, reference_date=ds)
    def get_candle_data(self, key, fd, td):
        return self.fetcher._fetch_single(key, "minutes", 5, fd, td)

class OptionChainProcessor:
    def __init__(self, strategy: MarketDataStrategy):
        self.strategy = strategy
    def process_data(self, df, spot_map, instr):
        if df is None or df.empty: return []
        df = df.rename(columns={'close': 'ltp', 'open_interest': 'oi'})
        df['change_in_oi'] = df['oi'].diff().fillna(0)
        strike, expiry, opt_type, r = instr['strike'], instr['expiry'], instr['type'], 0.1
        iv_list = []
        for idx, row in df.iterrows():
            spot_p = spot_map.get(idx)
            if spot_p:
                exp_t = expiry.replace(hour=23, minute=30) 
                if idx.tzinfo and not exp_t.tzinfo: exp_t = exp_t.replace(tzinfo=idx.tzinfo)
                T = (exp_t - idx).total_seconds() / (365 * 24 * 3600)
                iv_list.append(round(implied_volatility(row['ltp'], spot_p, strike, T, r, opt_type) * 100, 2) if T > 0 else 0)
            else: iv_list.append(0)
        df['iv'] = iv_list
        df['change_in_ltp'] = df['ltp'].diff().fillna(0)
        df['roc_oi'] = (df['oi'].pct_change() * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        df['roc_volume'] = (df['volume'].pct_change() * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        df['roc_iv'] = (df['iv'].pct_change() * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        df['coi_vol_ratio'] = (df['change_in_oi'] / df['volume']).replace([np.inf, -np.inf], 0).fillna(0).round(4)
        df['spot_price'] = df.index.map(spot_map).fillna(0)
        res = df[['ltp', 'change_in_ltp', 'roc_oi', 'roc_volume', 'roc_iv', 'coi_vol_ratio', 'spot_price']].reset_index()
        # MCX time filtering (09:00 - 23:30)
        res = res[(res['date'].dt.time >= datetime.strptime("09:00", "%H:%M").time())]
        res['date'] = res['date'].astype(str)
        return res.to_dict(orient='records')

    def run(self, ds, ts=None):
        try:
            sp = self.strategy.get_spot_price(ds, ts)
            if not sp: return self.save_meta(None, ds, ts, None, False, error="No spot price")
            instrs, exp, _ = self.strategy.get_instruments(sp, ds)
            if not instrs: return self.save_meta(sp, ds, ts, None, False, error="No instruments")
            
            # Fetch spot data for IV mapping
            if isinstance(self.strategy, LiveStrategy):
                sm_df = self.strategy.fetcher.fetch(get_spot_key(), "minutes", 5)
            else:
                sm_df = self.strategy.fetcher._fetch_single(get_spot_key(), "minutes", 5, ds, ds)
            
            spot_map = sm_df['close'].to_dict() if sm_df is not None else {}
            data = []
            for i in instrs:
                df = self.strategy.get_candle_data(i['key'], ds, ds)
                recs = self.process_data(df, spot_map, i)
                for r in recs:
                    r['symbol'] = i['symbol']
                    data.append(r)
            self.save_results(data, sp, ds, ts, exp)
        except Exception as e:
            print(f"Execution Error: {e}")
            self.save_meta(None, ds, ts, None, False, error=str(e))

    def save_meta(self, sp, ds, ts, exp, has_data, error=None):
        suffix = f"_{ts.replace(':', '')}" if ts else ""
        meta_file = f"mcx_meta_{ds}{suffix}.json"
        meta = {"spot_price": sp, "target_date": ds, "target_time": ts, "expiry_date": exp.strftime('%Y-%m-%d') if exp else None, "fetched_at": datetime.now().isoformat(), "has_data": has_data, "error": error}
        with open(meta_file, 'w') as f: json.dump(meta, f, indent=4)

    def save_results(self, data, sp, ds, ts, exp):
        suffix = f"_{ts.replace(':', '')}" if ts else ""
        if data:
            pd.DataFrame(data).to_csv(f"mcx_data_tabular_{ds}{suffix}.csv", index=False)
            self.save_meta(sp, ds, ts, exp, True)
        else: self.save_meta(sp, ds, ts, exp, False, error="No tabular data")

if __name__ == "__main__":
    live_mode = "--live" in sys.argv
    date_str = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else datetime.now().strftime('%Y-%m-%d')
    time_str = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    strategy = LiveStrategy() if (live_mode or date_str == datetime.now().strftime('%Y-%m-%d')) else HistoricalStrategy()
    OptionChainProcessor(strategy).run(date_str, time_str)
