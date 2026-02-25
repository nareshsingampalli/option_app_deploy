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
    if T <= 0 or sigma <= 0: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(price, S, K, T, r, option_type):
    """Calculate Implied Volatility using Newton-Raphson method."""
    if T <= 0: return 0
    sigma = 0.5
    for i in range(100):
        if option_type == 'CE':
            p = black_scholes_call(S, K, T, r, sigma)
        else:
            p = black_scholes_put(S, K, T, r, sigma)
        
        diff = p - price
        if abs(diff) < 0.0001:
            return sigma
        
        # Vega calculation
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 0.0001:
            break
        
        sigma = sigma - diff / vega
        if sigma <= 0:
            sigma = 0.001
            break
            
    return sigma

def get_expired_option_chain_instruments(spot_price, num_strikes=3, reference_date=None):
    """Fetch expired Nifty 50 option chain and select CE and PE around spot price."""
    print(f"Fetching Expired Option Chain for Spot: {spot_price} on {reference_date}...")
    fetcher = ExpiredCandleFetcher()
    underlying = get_nifty50_spot_key()
    
    # 1. Get expiries
    expiries = fetcher.fetch_expiries(underlying)
    
    # 2. Find target expiry
    target_date = datetime.strptime(reference_date, '%Y-%m-%d') if reference_date else datetime.now()
    target_expiry_str = None
    
    if expiries:
        # Find the first expiry >= target_date
        for exp in expiries:
            exp_dt = datetime.strptime(str(exp), '%Y-%m-%d')
            if exp_dt.date() >= target_date.date():
                target_expiry_str = str(exp)
                break
                
        if not target_expiry_str:
            print(f"No valid expiry found for date {target_date.date()}. Using last available: {expiries[-1]}")
            target_expiry_str = str(expiries[-1])
    else:
        print("No expiries found from API. Calculating Next Tuesday for Nifty 50 expiry fallback.")
        day_diff = 1 - target_date.weekday() # Tuesday is 1
        if day_diff < 0:
            day_diff += 7
        target_expiry_str = (target_date + timedelta(days=day_diff)).strftime('%Y-%m-%d')
        
    print(f"Target Expiry: {target_expiry_str}")
    
    # 3. Fetch contracts
    target_expiry_dt = datetime.strptime(target_expiry_str, '%Y-%m-%d')
    # Use string format for API compatibility
    contracts = fetcher.fetch_contracts(underlying, target_expiry_str)
    if not contracts:
        print("No contracts found for this expiry.")
        return [], target_expiry_dt
        
    df = pd.DataFrame([c.to_dict() if hasattr(c, 'to_dict') else c for c in contracts])
    
    # 4. Filter by strikes
    # Use direct attributes from expired API
    df['strike'] = df['strike_price'].astype(float)
    df['type'] = df['instrument_type']
    
    # Select closest strikes
    strikes = sorted(df['strike'].unique())
    # find nearest strike
    idx = np.abs(np.array(strikes) - spot_price).argmin()
    
    selected_strikes = strikes[max(0, idx-num_strikes):min(len(strikes), idx+num_strikes+1)]
    
    final_instruments = []
    for strike in selected_strikes:
        try:
            ce_rows = df[(df['strike'] == strike) & (df['type'] == 'CE')]
            pe_rows = df[(df['strike'] == strike) & (df['type'] == 'PE')]
            
            if not ce_rows.empty and not pe_rows.empty:
                ce = ce_rows.iloc[0]
                pe = pe_rows.iloc[0]
                
                final_instruments.append({
                    'key': ce['instrument_key'],
                    'symbol': ce['trading_symbol'],
                    'strike': strike,
                    'type': 'CE',
                    'expiry': target_expiry_dt
                })
                final_instruments.append({
                    'key': pe['instrument_key'],
                    'symbol': pe['trading_symbol'],
                    'strike': strike,
                    'type': 'PE',
                    'expiry': target_expiry_dt
                })
        except:
            continue
        
    return final_instruments, target_expiry_dt

def get_option_chain_instruments(spot_price, num_strikes=3, reference_date=None):
    """Fetch Nifty 50 option chain and select CE and PE around spot price."""
    print(f"Fetching Active Option Chain for Spot: {spot_price} (Ref: {reference_date})...")
    
    # Use the Assets URL which is more reliable than the V2 API endpoint
    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching instruments: {e}")
        return [], None, False
        
    content = gzip.decompress(response.content).decode('utf-8')
    df = pd.read_csv(io.StringIO(content))
    
    # Filter for Nifty 50 Options
    # The assets CSV uses 'name' and 'instrument_type' for filtering
    nifty_df = df[
        (df['instrument_type'] == 'OPTIDX') & 
        (df['name'] == 'NIFTY')
    ].copy()
    
    # Rename columns to match existing logic expectation
    nifty_df = nifty_df.rename(columns={
        'tradingsymbol': 'trading_symbol',
        'strike': 'strike_price'
    })
    
    if nifty_df.empty:
        return [], None, False
        
    nifty_df['expiry'] = pd.to_datetime(nifty_df['expiry'])
    
    # Find nearest expiry after reference_date
    ref_dt = pd.to_datetime(reference_date) if reference_date else pd.to_datetime(datetime.now().date())
    
    available_expiries = sorted(nifty_df['expiry'].unique())
    target_expiry = None
    
    for exp in available_expiries:
        if exp >= ref_dt:
            target_expiry = exp
            break
            
    if target_expiry is None:
        return [], None, True # Likely expired
        
    expiry_df = nifty_df[nifty_df['expiry'] == target_expiry].copy()
    
    # Find nearest strikes
    strikes = sorted(expiry_df['strike_price'].unique())
    idx = np.abs(np.array(strikes) - spot_price).argmin()
    
    selected_strikes = strikes[max(0, idx-num_strikes):min(len(strikes), idx+num_strikes+1)]
    
    final_instruments = []
    for strike in selected_strikes:
        ce = expiry_df[(expiry_df['strike_price'] == strike) & (expiry_df['option_type'] == 'CE')].iloc[0]
        pe = expiry_df[(expiry_df['strike_price'] == strike) & (expiry_df['option_type'] == 'PE')].iloc[0]
        
        final_instruments.append({
            'key': ce['instrument_key'],
            'symbol': ce['trading_symbol'],
            'strike': strike,
            'type': 'CE',
            'expiry': target_expiry
        })
        final_instruments.append({
            'key': pe['instrument_key'],
            'symbol': pe['trading_symbol'],
            'strike': strike,
            'type': 'PE',
            'expiry': target_expiry
        })
        
    return final_instruments, target_expiry, False

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
        # Try 5-minute first
        spot_df = self.fetcher.fetch(self.nifty_key, "minutes", 5)
        
        # Fallback to 1-minute if market just opened (09:15 - 09:20)
        if spot_df is None or spot_df.empty:
            print("5-minute data not yet available. Falling back to 1-minute data...")
            spot_df = self.fetcher.fetch(self.nifty_key, "minutes", 1)
            
        if spot_df is not None and not spot_df.empty:
            price = spot_df['close'].iloc[-1]
            print(f"Live Spot Price: {price}")
            return price
        return None
        
    def get_instruments(self, spot_price, target_date_str):
        return get_option_chain_instruments(spot_price, num_strikes=3, reference_date=target_date_str)
        
    def get_iv_spot_data(self, target_date_str):
        df = self.fetcher.fetch(self.nifty_key, "minutes", 5)
        if df is None or df.empty:
            df = self.fetcher.fetch(self.nifty_key, "minutes", 1)
        return df
        
    def get_candle_data(self, instrument, from_date, to_date):
        instr_key = instrument['key']
        df = self.fetcher.fetch(instr_key, "minutes", 5)
        if df is None or df.empty:
            df = self.fetcher.fetch(instr_key, "minutes", 1)
        return df

class HistoricalStrategy(MarketDataStrategy):
    """Strategy for fetching Historical data (Active historicals after last_expiry)."""
    
    def __init__(self):
        self.fetcher = HistoricalCandleFetcher()
        self.nifty_key = get_nifty50_spot_key()
        
    def get_spot_price(self, target_date_str, target_time_str):
        # Logic to fetch historical spot price
        spot_price = None
        target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')

        if target_time_str:
             print(f"Fetching Nifty 50 Spot data (5min) for {target_date_str} to find price at {target_time_str}...")
             spot_df_intra = self.fetcher._fetch_single(self.nifty_key, "minutes", 5, target_date_str, target_date_str)
             
             # Fallback for today's market open
             if (spot_df_intra is None or spot_df_intra.empty) and target_date_str == datetime.now().strftime('%Y-%m-%d'):
                print("5-minute data not yet available for today. Checking 1-minute data...")
                spot_df_intra = self.fetcher._fetch_single(self.nifty_key, "minutes", 1, target_date_str, target_date_str)

             if spot_df_intra is not None and not spot_df_intra.empty:
                target_full_dt = datetime.strptime(f"{target_date_str} {target_time_str}", "%Y-%m-%d %H:%M")
                try:
                    nearest_idx = spot_df_intra.index.get_indexer([target_full_dt], method='nearest')[0]
                    spot_price = spot_df_intra.iloc[nearest_idx]['close']
                    print(f"Found nearest spot candle at {spot_df_intra.index[nearest_idx]}: {spot_price}")
                except:
                    pass

        if spot_price is None:
            # Strictly fetch Daily for the target date
            print(f"Fetching Nifty 50 Spot data (Daily) for {target_date_str}...")
            # We fetch a range around the target date to handle holidays/weekends
            spot_df_daily = self.fetcher.fetch(self.nifty_key, timeframe="days", lookback_days=15)
            
            if spot_df_daily is not None and not spot_df_daily.empty:
                # Filter up to target_date_str
                available_dates = spot_df_daily.index.strftime('%Y-%m-%d')
                if target_date_str in available_dates:
                    spot_price = spot_df_daily.loc[target_date_str]['close']
                    if isinstance(spot_price, pd.Series): spot_price = spot_price.iloc[0]
                    print(f"Nifty 50 Spot Close on {target_date_str}: {spot_price}")
                else:
                    # Find the nearest previous closing price
                    target_dt_aware = target_dt
                    if spot_df_daily.index.tzinfo is not None:
                         # Use the timezone from the index directly to avoid pytz dependency if possible
                         # but keep the aware logic
                         target_dt_aware = target_dt.replace(tzinfo=spot_df_daily.index.tzinfo)
                    
                    past_data = spot_df_daily[spot_df_daily.index <= target_dt_aware]
                    if not past_data.empty:
                        spot_price = past_data['close'].iloc[-1]
                        print(f"Target date {target_date_str} not found. Using nearest previous Close ({past_data.index[-1].date()}): {spot_price}")
                    
        return spot_price
        
    def get_instruments(self, spot_price, target_date_str):
        # Standard instruments for non-expired historical
        instruments, expiry, is_expired = get_option_chain_instruments(spot_price, num_strikes=3, reference_date=target_date_str)
        return instruments, expiry, is_expired
        
    def get_iv_spot_data(self, target_date_str):
        df = self.fetcher._fetch_single(self.nifty_key, "minutes", 5, target_date_str, target_date_str)
        if (df is None or df.empty) and target_date_str == datetime.now().strftime('%Y-%m-%d'):
            df = self.fetcher._fetch_single(self.nifty_key, "minutes", 1, target_date_str, target_date_str)
        return df
        
    def get_candle_data(self, instrument, from_date, to_date):
        instr_key = instrument['key']
        df = self.fetcher._fetch_single(instr_key, "minutes", 5, to_date, from_date)
        if (df is None or df.empty) and to_date == datetime.now().strftime('%Y-%m-%d'):
            df = self.fetcher._fetch_single(instr_key, "minutes", 1, to_date, from_date)
        return df

class ExpiredStrategy(MarketDataStrategy):
    """Strategy for fetching data for recently Expired contracts."""
    
    def __init__(self):
        self.fetcher = ExpiredCandleFetcher()
        self.hist_fetcher = HistoricalCandleFetcher()
        self.nifty_key = get_nifty50_spot_key()
        
    def get_spot_price(self, target_date_str, target_time_str):
        # Logic to fetch expired spot price
        spot_price = None
        target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')

        if target_time_str:
             print(f"Fetching Nifty 50 Spot data (5min) for {target_date_str} at {target_time_str}...")
             # Indices are NOT in the Expired API, use HistoricalCandleFetcher
             spot_df_intra = self.hist_fetcher._fetch_single(self.nifty_key, "minutes", 5, target_date_str, target_date_str)
             
             if spot_df_intra is not None and not spot_df_intra.empty:
                target_full_dt = datetime.strptime(f"{target_date_str} {target_time_str}", "%Y-%m-%d %H:%M")
                try:
                    nearest_idx = spot_df_intra.index.get_indexer([target_full_dt], method='nearest')[0]
                    spot_price = spot_df_intra.iloc[nearest_idx]['close']
                    print(f"Found nearest spot candle at {spot_df_intra.index[nearest_idx]}: {spot_price}")
                except:
                    pass

        if spot_price is None:
            # Strictly fetch Daily for the target date
            print(f"Fetching Nifty 50 Spot data (Daily) for {target_date_str}...")
            # Indices are NOT in the Expired API, use HistoricalCandleFetcher
            spot_df_daily = self.hist_fetcher.fetch(self.nifty_key, timeframe="days", lookback_days=30)
            
            if spot_df_daily is not None and not spot_df_daily.empty:
                available_dates = spot_df_daily.index.strftime('%Y-%m-%d')
                if target_date_str in available_dates:
                    spot_price = spot_df_daily.loc[target_date_str]['close']
                    if isinstance(spot_price, pd.Series): spot_price = spot_price.iloc[0]
                    print(f"Nifty 50 Spot Close on {target_date_str}: {spot_price}")
                else:
                    # Find the nearest previous closing price
                    target_dt_aware = target_dt
                    if spot_df_daily.index.tzinfo is not None:
                         target_dt_aware = target_dt.replace(tzinfo=spot_df_daily.index.tzinfo)

                    past_data = spot_df_daily[spot_df_daily.index <= target_dt_aware]
                    if not past_data.empty:
                        spot_price = past_data['close'].iloc[-1]
                        print(f"Target date {target_date_str} not found. Using nearest previous Close ({past_data.index[-1].date()}): {spot_price}")
                        
        return spot_price
        
    def get_instruments(self, spot_price, target_date_str):
        # Force fetch from expired instruments
        instruments, expiry = get_expired_option_chain_instruments(spot_price, num_strikes=3, reference_date=target_date_str)
        return instruments, expiry, True
        
    def get_iv_spot_data(self, target_date_str):
        return self.hist_fetcher._fetch_single(self.nifty_key, "minutes", 5, target_date_str, target_date_str)
        
    def _format_expired_key(self, key, expiry_dt):
        if not key or not expiry_dt: return key
        if key.count('|') >= 2: return key
        return f"{key}|{expiry_dt.strftime('%d-%m-%Y')}"

    def get_candle_data(self, instrument, from_date, to_date):
        instr_key = instrument['key']
        exp_key = self._format_expired_key(instr_key, instrument.get('expiry'))
        return self.fetcher.fetch_candle_data(exp_key, "5minute", to_date, from_date)

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
        df['spot_price'] = df['spot_price'].replace(0, pd.NA).ffill().bfill().fillna(0)

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
        """Fetch spot, select instruments, fetch option candles, process, and save."""
        # 1. Fetch Spot
        spot_price = self.strategy.get_spot_price(target_date_str, target_time_str)
        if spot_price is None:
            print("Failed to fetch spot price. Exiting.")
            return

        # 2. Get Instruments
        instruments, target_expiry, is_expired = self.strategy.get_instruments(spot_price, target_date_str)
        if not instruments:
            print(f"No instruments found for {target_date_str}. Saving meta only.")
            self.save_meta(spot_price, target_date_str, target_time_str, target_expiry, False)
            return

        # 3. Create Spot Map for IV calculation
        spot_df = self.strategy.get_iv_spot_data(target_date_str)
        spot_map = {}
        if spot_df is not None and not spot_df.empty:
            spot_map = spot_df['close'].to_dict()

        # 4. Fetch and Process each instrument
        results = []
        for inst in instruments:
            print(f"Fetching candles for {inst['symbol']}...")
            try:
                # Candle range: from target date back 5 days to handle ROC on first session candles
                to_date = target_date_str
                from_date = (datetime.strptime(target_date_str, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
                
                df_candles = self.strategy.get_candle_data(inst, from_date, to_date)
                if df_candles is not None and not df_candles.empty:
                    processed = self.process_data(df_candles, spot_map, inst)
                    for row in processed:
                        row['symbol'] = inst['symbol']
                        results.append(row)
            except Exception as e:
                print(f"Error processing {inst['symbol']}: {e}")

        # 5. Save results
        if results:
            self.save_results(results, spot_price, target_date_str, target_time_str, target_expiry, is_expired)
            self.save_meta(spot_price, target_date_str, target_time_str, target_expiry, True, expired_error=is_expired)
        else:
            print("No data available to save.")
            # Check if it was an expiry issue
            is_expired_error = is_expired
            if spot_price and not target_expiry:
                is_expired_error = True
            self.save_meta(spot_price, target_date_str, target_time_str, target_expiry, False, is_expired_error)

    def save_results(self, data, spot_price, target_date_str, target_time_str, target_expiry, is_expired=False):
        df = pd.DataFrame(data)
        if target_time_str:
            clean_time = target_time_str.replace(":", "")
            filename = f"option_data_tabular_{target_date_str}_{clean_time}.csv"
        else:
            filename = f"option_data_tabular_{target_date_str}.csv"
            
        df.to_csv(filename, index=False)
        print(f"Saved results to {filename}")

    def save_meta(self, spot_price, target_date_str, target_time_str, target_expiry, has_data, expired_error=False):
        meta = {
            "spot_price": spot_price,
            "target_date": target_date_str,
            "target_time": target_time_str,
            "expiry_date": str(target_expiry.date()) if target_expiry else None,
            "fetched_at": datetime.now().isoformat(),
            "has_data": has_data,
            "expired_contracts": expired_error
        }
        if target_time_str:
            clean_time = target_time_str.replace(":", "")
            filename = f"option_meta_{target_date_str}_{clean_time}.json"
        else:
            filename = f"option_meta_{target_date_str}.json"
            
        with open(filename, 'w') as f:
            json.dump(meta, f, indent=4)
        print(f"Saved metadata to {filename}")


def main():
    live_mode = False
    if "--live" in sys.argv:
        live_mode = True
        
    # Check for date argument
    today_str = datetime.now().strftime('%Y-%m-%d')
    target_date_str = today_str
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_date_str = sys.argv[1]
        
    # Check for time argument
    target_time_str = None
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        target_time_str = sys.argv[2]
        
    if live_mode or (target_date_str == today_str and not target_time_str):
        print(f"Running in LIVE MODE for {target_date_str}")
        strategy = LiveStrategy()
    else:
        # Determine if we need Historical or Expired strategy
        target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        last_expiry = None
        try:
            # Quick check for last_expiry using a temporary fetcher
            temp_fetcher = ExpiredCandleFetcher()
            expiries = temp_fetcher.fetch_expiries(get_nifty50_spot_key())
            if expiries:
                last_expiry = datetime.strptime(str(expiries[-1]), '%Y-%m-%d')
        except:
            pass
            
        if last_expiry and target_dt <= last_expiry:
            print(f"Date {target_date_str} is Expired (Last Expiry: {last_expiry.date()}). Using ExpiredStrategy.")
            strategy = ExpiredStrategy()
        else:
            if target_date_str == today_str:
                print(f"Running in TIME-BASED INTRADAY MODE for {target_date_str} at {target_time_str}")
            else:
                print(f"Running in HISTORICAL MODE for {target_date_str}")
            strategy = HistoricalStrategy()
        
    processor = OptionChainProcessor(strategy)
    processor.run(target_date_str, target_time_str)

if __name__ == "__main__":
    main()
