import os
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
# Also try loading from refactor_app .env (common location on VM)
load_dotenv("/home/ubuntu/refactor_app/.env")

def get_nifty50_spot_key():
    """Get the instrument key for Nifty 50 Index."""
    return "NSE_INDEX|Nifty 50"

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(price, S, K, T, r, option_type):
    """Calculate Implied Volatility using Newton-Raphson method."""
    sigma = 0.5  # Initial guess
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
            
            if vega == 0:
                break
            
            sigma = sigma + diff / vega
            
            # Clamp sigma to reasonable values
            if sigma <= 0: sigma = 0.001
            if sigma > 5: sigma = 5
            
        return sigma
    except:
        return 0

def get_expired_option_chain_instruments(spot_price, num_strikes=3, reference_date=None):
    """Fetch expired Nifty 50 option chain and select CE and PE around spot price."""
    print(f"Fetching Expired Nifty 50 Option Chain for Spot: {spot_price}")
    fetcher = ExpiredCandleFetcher()
    
    try:
        nifty_key = get_nifty50_spot_key()
        print("Fetching expired expiries...")
        expiries = fetcher.fetch_expiries(nifty_key)
        
        if not expiries:
            print("No expired expiries found.")
            return [], None
            
        expiry_dates = []
        for e in expiries:
            try:
                dt = datetime.strptime(str(e), '%Y-%m-%d')
                expiry_dates.append(dt)
            except:
                pass
        expiry_dates.sort()
        
        if not reference_date:
            reference_date = datetime.now()
            
        ref_dt = pd.to_datetime(reference_date)
        target_expiry = None
        for exp in expiry_dates:
            if exp >= ref_dt:
                target_expiry = exp
                break
                
        if not target_expiry:
            print(f"No suitable expired expiry found after {ref_dt.date()}.")
            return [], None
            
        print(f"Target Expired Expiry: {target_expiry.date()}")
        print(f"Fetching contracts for expiry {target_expiry.date()}...")
        contracts = fetcher.fetch_contracts(nifty_key, target_expiry.strftime('%Y-%m-%d'))
        
        if not contracts:
            print("No contracts found for this expiry.")
            return [], None
            
        contracts_data = []
        for c in contracts:
            if hasattr(c, 'to_dict'):
                contracts_data.append(c.to_dict())
            elif isinstance(c, dict):
                contracts_data.append(c)
            else:
                try: contracts_data.append(c.__dict__)
                except: pass
        
        if not contracts_data:
            return [], None

        df = pd.DataFrame(contracts_data)
        
        if 'instrument_type' not in df.columns:
            return [], None
            
        if 'underlying_symbol' in df.columns:
            df = df[df['underlying_symbol'] == 'NIFTY']
        elif 'name' in df.columns:
            df = df[df['name'] == 'NIFTY']
        
        df = df[df['instrument_type'].isin(['CE', 'PE'])]
        
        df['strike'] = df['strike_price'].astype(float)
        df['dist'] = abs(df['strike'] - spot_price)
        
        ce_opts = df[df['instrument_type'] == 'CE'].sort_values('dist')
        pe_opts = df[df['instrument_type'] == 'PE'].sort_values('dist')
        
        selected_ce = ce_opts.head(num_strikes)
        selected_pe = pe_opts.head(num_strikes)
        
        selected_instruments = []
        for _, row in selected_ce.iterrows():
            selected_instruments.append({
                'symbol': row['trading_symbol'],
                'key': row['instrument_key'],
                'type': 'CE',
                'strike': row['strike'],
                'expiry': target_expiry
            })
            
        for _, row in selected_pe.iterrows():
            selected_instruments.append({
                'symbol': row['trading_symbol'],
                'key': row['instrument_key'],
                'type': 'PE',
                'strike': row['strike'],
                'expiry': target_expiry
            })
            
        return selected_instruments, target_expiry
        
    except Exception as e:
        # Re-raise critical exceptions like Auth/RateLimit to allow upper layers to handle them
        if "401" in str(e) or "Unauthorized" in str(e) or "429" in str(e) or "Invalid token" in str(e):
             print(f"Critical error in get_expired_option_chain_instruments: {e}")
             raise e
        print(f"Error fetching option chain: {e}")
        return [], None

def get_option_chain_instruments(spot_price, num_strikes=3, reference_date=None):
    """Fetch Nifty 50 option chain and select CE and PE around spot price."""
    print(f"Fetching Nifty 50 Option Chain for Spot: {spot_price}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        
        with gzip.open(io.BytesIO(r.content), 'rb') as f:
            df = pd.read_json(f)
            
        nifty_opts = df[
            (df['instrument_type'].isin(['CE', 'PE'])) & 
            (df['underlying_symbol'] == 'NIFTY')
        ]
        
        if nifty_opts.empty:
             nifty_opts = df[
                (df['instrument_type'].isin(['CE', 'PE'])) & 
                (df['trading_symbol'].str.startswith('NIFTY')) &
                (~df['trading_symbol'].str.contains('BANK')) &
                (~df['trading_symbol'].str.contains('FIN'))
            ]

        nifty_opts = nifty_opts.copy()
        nifty_opts['expiry_dt'] = pd.to_datetime(nifty_opts['expiry'], unit='ms')
        
        if not reference_date:
            reference_date = datetime.now()
        
        ref_dt = pd.to_datetime(reference_date)
        valid_expiries = nifty_opts[nifty_opts['expiry_dt'] >= ref_dt]['expiry_dt'].sort_values().unique()
        
        should_use_expired = False
        target_expiry = None
        
        if len(valid_expiries) == 0:
            print(f"No future expiries found after {ref_dt.date()}. Checking expired instruments...")
            should_use_expired = True
        else:
            target_expiry = valid_expiries[0]
            expiry_diff = (target_expiry - ref_dt).days
            if expiry_diff > 10:
                print(f"Warning: Nearest available expiry ({target_expiry.date()}) is {expiry_diff} days away.")
                print("Checking expired instruments...")
                should_use_expired = True
        
        if should_use_expired:
            instruments, expired_expiry = get_expired_option_chain_instruments(spot_price, num_strikes, reference_date)
            if instruments:
                return instruments, expired_expiry, True
            else:
                return [], None, False

        print(f"Target Expiry: {target_expiry.date()}")
        
        opts_expiry = nifty_opts[nifty_opts['expiry_dt'] == target_expiry].copy()
        opts_expiry['strike'] = opts_expiry['strike_price'].astype(float)
        opts_expiry['dist'] = abs(opts_expiry['strike'] - spot_price)
        
        ce_opts = opts_expiry[opts_expiry['instrument_type'] == 'CE'].sort_values('dist')
        pe_opts = opts_expiry[opts_expiry['instrument_type'] == 'PE'].sort_values('dist')
        
        selected_ce = ce_opts.head(num_strikes)
        selected_pe = pe_opts.head(num_strikes)
        
        selected_instruments = []
        for _, row in selected_ce.iterrows():
            selected_instruments.append({
                'symbol': row['trading_symbol'],
                'key': row['instrument_key'],
                'type': 'CE',
                'strike': row['strike'],
                'expiry': target_expiry
            })
            
        for _, row in selected_pe.iterrows():
            selected_instruments.append({
                'symbol': row['trading_symbol'],
                'key': row['instrument_key'],
                'type': 'PE',
                'strike': row['strike'],
                'expiry': target_expiry
            })
            
        return selected_instruments, target_expiry, False

    except Exception as e:
        print(f"Error fetching option chain: {e}")
        raise e

# --- Strategy Design Pattern ---

class MarketDataStrategy(ABC):
    """Abstract Strategy for fetching market data (Spot, Instruments, Candles)."""
    
    @abstractmethod
    def get_spot_price(self, target_date_str, target_time_str):
        """Fetch the spot price for instrument selection."""
        pass
        
    @abstractmethod
    def get_instruments(self, spot_price, target_date_str):
        """Fetch relevant option instruments. Returns (instruments, expiry, is_expired)."""
        pass
        
    @abstractmethod
    def get_iv_spot_data(self, target_date_str):
        """Fetch spot data (5-min) for IV calculation."""
        pass
        
    @abstractmethod
    def get_candle_data(self, instrument_key, from_date, to_date):
        """Fetch candle data for the option instrument."""
        pass

class LiveStrategy(MarketDataStrategy):
    """Strategy for fetching Live/Intraday data."""
    
    def __init__(self):
        self.fetcher = IntradayCandleFetcher()
        self.nifty_key = get_nifty50_spot_key()
        
    def get_spot_price(self, target_date_str, target_time_str):
        print("Fetching Live Nifty 50 Spot data (Intraday)...")
        spot_df = self.fetcher.fetch(self.nifty_key, "minutes", 5)
        if spot_df is not None and not spot_df.empty:
            price = spot_df['close'].iloc[-1]
            print(f"Live Spot Price: {price}")
            return price
        return None
        
    def get_instruments(self, spot_price, target_date_str):
        # For live, we assume standard active instruments
        # We pass reference_date as today/target_date
        return get_option_chain_instruments(spot_price, num_strikes=5, reference_date=target_date_str)
        
    def get_iv_spot_data(self, target_date_str):
        return self.fetcher.fetch(self.nifty_key, "minutes", 5)
        
    def get_candle_data(self, instrument_key, from_date, to_date):
        return self.fetcher.fetch(instrument_key, "minutes", 5)

class HistoricalStrategy(MarketDataStrategy):
    """Strategy for fetching Historical data (handles both Active and Expired historicals)."""
    
    def __init__(self):
        self.fetcher = HistoricalCandleFetcher()
        self.expired_fetcher = ExpiredCandleFetcher()
        self.nifty_key = get_nifty50_spot_key()
        self.is_expired_mode = False # State to track if we switched to expired
        
    def get_spot_price(self, target_date_str, target_time_str):
        # Logic to fetch historical spot price
        spot_price = None
        
        # Determine if we need to use the expired fetcher based on dynamic detection of last_expiry
        try:
            expiries = self.expired_fetcher.fetch_expiries(self.nifty_key)
            if expiries:
                last_expiry_str = str(expiries[-1])
                last_expiry = datetime.strptime(last_expiry_str, '%Y-%m-%d')
            else:
                last_expiry = datetime.now() - timedelta(days=365)
        except Exception as e:
            print(f"Error detecting last_expiry for spot: {e}")
            last_expiry = datetime.now() - timedelta(days=365)

        target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        use_expired = target_dt <= last_expiry

        if target_time_str:
             if use_expired:
                 print(f"Fetching Expired Nifty 50 Spot data (5min) for {target_date_str} at {target_time_str}...")
                 spot_df_intra = self.expired_fetcher.fetch_candle_data(self.nifty_key, "5minute", target_date_str, target_date_str)
             else:
                 print(f"Fetching Nifty 50 Spot data (5min) for {target_date_str} to find price at {target_time_str}...")
                 spot_df_intra = self.fetcher._fetch_single(self.nifty_key, "minutes", 5, target_date_str, target_date_str)
             
             if spot_df_intra is not None and not spot_df_intra.empty:
                target_full_dt = datetime.strptime(f"{target_date_str} {target_time_str}", "%Y-%m-%d %H:%M")
                try:
                    nearest_idx = spot_df_intra.index.get_indexer([target_full_dt], method='nearest')[0]
                    spot_price = spot_df_intra.iloc[nearest_idx]['close']
                    print(f"Found nearest spot candle at {spot_df_intra.index[nearest_idx]}: {spot_price}")
                except:
                    pass

        if spot_price is None:
            # Fallback to Daily
            today = datetime.now()
            days_diff = (today - target_dt).days + 5
            lookback = max(5, days_diff)
            
            if use_expired:
                print(f"Fetching Expired Nifty 50 Spot data (Daily) for {target_date_str}...")
                from_date = (target_dt - timedelta(days=lookback)).strftime('%Y-%m-%d')
                spot_df_daily = self.expired_fetcher.fetch_candle_data(self.nifty_key, "day", target_date_str, from_date)
            else:
                print(f"Fetching Nifty 50 Spot data (Daily) with lookback {lookback} days...")
                spot_df_daily = self.fetcher.fetch(self.nifty_key, timeframe="days", lookback_days=lookback)
            
            if spot_df_daily is not None and not spot_df_daily.empty:
                if target_date_str in spot_df_daily.index.strftime('%Y-%m-%d'):
                    spot_price = spot_df_daily.loc[target_date_str]['close']
                    if isinstance(spot_price, pd.Series): spot_price = spot_price.iloc[0]
                    print(f"Nifty 50 Spot Close on {target_date_str}: {spot_price}")
                else:
                    spot_price = spot_df_daily['close'].iloc[-1]
                    print(f"Using Latest Spot Close: {spot_price}")
                    
        return spot_price
        
    def get_instruments(self, spot_price, target_date_str):
        instruments, expiry, is_expired = get_option_chain_instruments(spot_price, num_strikes=5, reference_date=target_date_str)
        self.is_expired_mode = is_expired
        if is_expired:
            print("HistoricalStrategy: Switched to Expired Mode.")
        return instruments, expiry, is_expired
        
    def get_iv_spot_data(self, target_date_str):
        # Determine if we need to use the expired fetcher
        try:
            expiries = self.expired_fetcher.fetch_expiries(self.nifty_key)
            if expiries:
                last_expiry_str = str(expiries[-1])
                last_expiry = datetime.strptime(last_expiry_str, '%Y-%m-%d')
            else:
                last_expiry = datetime.now() - timedelta(days=365)
        except:
            last_expiry = datetime.now() - timedelta(days=365)

        target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        if target_dt <= last_expiry:
             return self.expired_fetcher.fetch_candle_data(self.nifty_key, "5minute", target_date_str, target_date_str)
        
        return self.fetcher._fetch_single(self.nifty_key, "minutes", 5, target_date_str, target_date_str)
        
    def _format_expired_key(self, key, expiry_dt):
        """Convert standard key to expired format: EXCHANGE_TYPE|ID|DD-MM-YYYY"""
        if not key or not expiry_dt: return key
        # If already formatted (contains 2 pipes), return as is
        if key.count('|') >= 2: return key
        return f"{key}|{expiry_dt.strftime('%d-%m-%Y')}"

    def get_candle_data(self, instrument, from_date, to_date):
        # Determine if we need to use the expired fetcher based on dynamic detection of last_expiry
        symbol = instrument.get('symbol', 'Unknown')
        instr_key = instrument['key']
        target_dt = datetime.strptime(to_date, '%Y-%m-%d')
        now = datetime.now()
        
        last_expiry = None
        try:
            # We use the Nifty Index key to get the general list of expired dates
            # (assuming standard expiries apply to all instruments in the chain)
            print(f"  -> Detecting last_expiry for {symbol}...")
            expired_dates_str = self.expired_fetcher.fetch_expiries(self.nifty_key)
            if expired_dates_str:
                expired_dates = []
                for d in expired_dates_str:
                    try:
                        expired_dates.append(datetime.strptime(str(d), '%Y-%m-%d'))
                    except:
                        pass
                if expired_dates:
                    last_expiry = max(expired_dates)
                    print(f"  -> Detected Last Expiry: {last_expiry.date()}")
        except Exception as e:
            print(f"  -> Warning: Failed to detect last_expiry: {e}")

        use_expired_fetcher = False
        
        # Rule 1: Explicitly flagged by get_instruments
        if self.is_expired_mode:
            use_expired_fetcher = True
        # Rule 2: Target Date is on or before Last Expiry (Expired Historical)
        elif last_expiry and target_dt <= last_expiry:
            use_expired_fetcher = True
            print(f"  -> Target {to_date} <= Last Expiry {last_expiry.date()}. Switching to Expired API.")
        # Rule 3: Safety check - if expiry is in past but not detected in list
        else:
            instr_expiry = instrument.get('expiry')
            if instr_expiry and instr_expiry.date() < now.date() and target_dt <= instr_expiry:
                use_expired_fetcher = True
                print(f"  -> Instrument expiry {instr_expiry.date()} is in the past. Switching to Expired API.")

        if use_expired_fetcher:
            # Format key if needed
            exp_key = self._format_expired_key(instr_key, instrument.get('expiry'))
            print(f"  -> Fetching using Expired Key: {exp_key}")
            return self.expired_fetcher.fetch_candle_data(exp_key, "5minute", to_date, from_date)
        else:
            return self.fetcher._fetch_single(instr_key, "minutes", 5, to_date, from_date)

class OptionChainProcessor:
    """Context class that executes the data processing workflow using a Strategy."""
    
    def __init__(self, strategy: MarketDataStrategy):
        self.strategy = strategy
        
    def process_data(self, df, spot_map, instr):
        """Process raw candle data to add indicators (IV, ROC, etc)."""
        if df is None or df.empty:
            return []
            
        # Rename columns
        df = df.rename(columns={'close': 'ltp', 'open_interest': 'oi'})
        
        # Calculate Change in OI
        df['change_in_oi'] = df['oi'].diff().fillna(0)
        
        # Calculate IV
        iv_list = []
        strike = instr['strike']
        expiry = instr['expiry']
        option_type = instr['type']
        r = 0.1 # Risk-free rate
        
        for idx, row in df.iterrows():
            spot_p = spot_map.get(idx)
            if spot_p:
                expiry_with_time = expiry.replace(hour=15, minute=30)
                if idx.tzinfo is not None and expiry_with_time.tzinfo is None:
                    expiry_with_time = expiry_with_time.replace(tzinfo=idx.tzinfo)
                
                t_delta = expiry_with_time - idx
                T = t_delta.total_seconds() / (365 * 24 * 3600)
                
                if T <= 0: iv = 0
                else: iv = implied_volatility(row['ltp'], spot_p, strike, T, r, option_type)
                iv_list.append(round(iv * 100, 2))
            else:
                iv_list.append(0)
        
        df['iv'] = iv_list
        df['change_in_ltp'] = df['ltp'].diff().fillna(0)
        
        # ROC calculations
        df['roc_oi'] = df['oi'].pct_change().replace([np.inf, -np.inf], 0).fillna(0) * 100
        df['roc_volume'] = df['volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0) * 100
        df['roc_iv'] = df['iv'].pct_change().replace([np.inf, -np.inf], 0).fillna(0) * 100
        
        # Rounding
        df['roc_oi'] = df['roc_oi'].round(2)
        df['roc_volume'] = df['roc_volume'].round(2)
        df['roc_iv'] = df['roc_iv'].round(2)
        
        # COI/Vol Ratio
        df['coi_vol_ratio'] = df.apply(lambda row: row['change_in_oi'] / row['volume'] if row['volume'] > 0 else 0, axis=1)
        df['coi_vol_ratio'] = df['coi_vol_ratio'].round(4)

        # Add Spot Price
        df['spot_price'] = df.index.map(spot_map)
        df['spot_price'] = df['spot_price'].fillna(0)

        # Select columns
        final_df = df[['ltp', 'change_in_ltp', 'roc_oi', 'roc_volume', 'roc_iv', 'coi_vol_ratio', 'spot_price']]
        final_df = final_df.reset_index()
        
        # Filter time range 09:15 - 15:15
        final_df = final_df[
            (final_df['date'].dt.time >= datetime.strptime("09:15", "%H:%M").time()) & 
            (final_df['date'].dt.time <= datetime.strptime("15:15", "%H:%M").time())
        ]
        
        final_df['date'] = final_df['date'].astype(str)
        return final_df.to_dict(orient='records')

    def run(self, target_date_str, target_time_str=None):
        try:
            print(f"Starting execution with {self.strategy.__class__.__name__}")
            
            # 1. Get Spot Price
            spot_price = self.strategy.get_spot_price(target_date_str, target_time_str)
            if not spot_price:
                print("Failed to fetch spot price.")
                self.save_meta(None, target_date_str, target_time_str, None, False, error="Failed to fetch spot price")
                return

            # 2. Get Instruments
            instruments, target_expiry, _ = self.strategy.get_instruments(spot_price, target_date_str)
            if not instruments:
                print("No instruments found.")
                # We still want to save meta if possible?
                self.save_meta(spot_price, target_date_str, target_time_str, None, False, error="No instruments found")
                return

            print(f"Selected {len(instruments)} instruments")
            
            # 3. Get Spot Data for IV
            print("Fetching Spot Data for IV...")
            spot_df_iv = self.strategy.get_iv_spot_data(target_date_str)
            spot_map = {}
            if spot_df_iv is not None and not spot_df_iv.empty:
                spot_map = spot_df_iv['close'].to_dict()
            else:
                print("Warning: IV spot data missing. IV will be 0.")

            # 4. Process Instruments
            results = {}
            tabular_data = []
            
            for instr in instruments:
                print(f"Processing {instr['symbol']}...")
                df = self.strategy.get_candle_data(instr, target_date_str, target_date_str)
                records = self.process_data(df, spot_map, instr)
                
                if records:
                    print(f"  -> {len(records)} candles processed.")
                    results[instr['symbol']] = records
                    for r in records:
                        r['symbol'] = instr['symbol']
                        tabular_data.append(r)
                else:
                    print("  -> No data found.")
            
            # 5. Save Data
            self.save_results(tabular_data, spot_price, target_date_str, target_time_str, target_expiry)
        except Exception as e:
            print(f"Execution failed: {e}")
            self.save_meta(None, target_date_str, target_time_str, None, False, error=str(e))

    def save_meta(self, spot_price, target_date_str, target_time_str, target_expiry, has_data, expired_error=False, error=None):
        if target_time_str:
            clean_time = target_time_str.replace(":", "")
            meta_file = f"option_meta_{target_date_str}_{clean_time}.json"
        else:
            meta_file = f"option_meta_{target_date_str}.json"
            
        expiry_str = target_expiry.strftime('%Y-%m-%d') if target_expiry else None
        
        meta_data = {
            "spot_price": spot_price,
            "target_date": target_date_str,
            "target_time": target_time_str,
            "expiry_date": expiry_str,
            "fetched_at": datetime.now().isoformat(),
            "has_data": has_data,
            "expired_contracts": expired_error,
            "error": error
        }
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=4)
        print(f"Metadata stored in {meta_file}")

    def save_results(self, tabular_data, spot_price, target_date_str, target_time_str, target_expiry):
        if tabular_data:
            full_df = pd.DataFrame(tabular_data)
            full_df = full_df.sort_values(by=['date', 'symbol'])
            
            if target_time_str:
                clean_time = target_time_str.replace(":", "")
                csv_file = f"option_data_tabular_{target_date_str}_{clean_time}.csv"
            else:
                csv_file = f"option_data_tabular_{target_date_str}.csv"
                
            full_df.to_csv(csv_file, index=False)
            print(f"Tabular data stored in {csv_file}")
            
            self.save_meta(spot_price, target_date_str, target_time_str, target_expiry, True)
        else:
            print("No data available to save.")
            # Check if it was an expiry issue
            is_expired_error = False
            if spot_price and not target_expiry:
                is_expired_error = True
            self.save_meta(spot_price, target_date_str, target_time_str, target_expiry, False, is_expired_error)


def main():
    live_mode = False
    if "--live" in sys.argv:
        live_mode = True
        
    # Check for date argument
    target_date_str = "2026-02-20"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_date_str = sys.argv[1]
        
    # Check for time argument
    target_time_str = None
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        target_time_str = sys.argv[2]
        
    if live_mode:
        print("Running in LIVE MODE")
        target_date_str = datetime.now().strftime('%Y-%m-%d')
        target_time_str = None
        strategy = LiveStrategy()
    elif target_date_str == datetime.now().strftime('%Y-%m-%d'):
        print("Date is Today, switching to LIVE MODE (Intraday Data)...")
        live_mode = True
        target_time_str = None
        strategy = LiveStrategy()
    else:
        print(f"Running in HISTORICAL MODE. Date: {target_date_str}, Time: {target_time_str}")
        strategy = HistoricalStrategy()
        
    processor = OptionChainProcessor(strategy)
    processor.run(target_date_str, target_time_str)

if __name__ == "__main__":
    main()
