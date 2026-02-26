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
import sentry_sdk
from candle_fetchers import HistoricalCandleFetcher, ExpiredCandleFetcher, IntradayCandleFetcher

# Index instrument keys used for expired API lookups
INDEX_KEYS = {
    "NIFTY":     "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "FINNIFTY":  "NSE_INDEX|Nifty Fin Service",
    "SENSEX":    "BSE_INDEX|SENSEX",
}

# Load environment variables
load_dotenv()
# Also try loading from refactor_app .env (common location on VM)
load_dotenv("/home/ubuntu/refactor_app/.env")

# Initialize Sentry
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
    print(f"[Sentry] Monitoring enabled for {os.path.basename(__file__)}")

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

def get_expired_option_chain_instruments(spot_price, num_strikes=3, reference_date=None, symbol="NIFTY"):
    """
    Fetch expired option chain contracts and select CE+PE strikes nearest to spot_price.

    Flow:
      1. Resolve underlying index key via INDEX_KEYS.
      2. Fetch available expiry dates from the expired-instruments API.
      3. Find the nearest expiry >= reference_date (or last available if none found).
      4. Fetch all option contracts for that expiry.
      5. Select `num_strikes` strikes on each side of ATM (spot_price).
    """
    underlying = INDEX_KEYS.get(symbol.upper(), INDEX_KEYS["NIFTY"])
    print(f"[ExpiredFetcher] Symbol={symbol}, Underlying={underlying}, Spot={spot_price}, Date={reference_date}")

    fetcher = ExpiredCandleFetcher()

    # ── Step 1: Fetch available expiry dates ────────────────────────────────
    expiries = fetcher.fetch_expiries(underlying)
    if not expiries:
        print("[ExpiredFetcher] No expiries returned from API.")
        return [], None

    print(f"[ExpiredFetcher] Available expiries ({len(expiries)}): {expiries[:5]}...")

    # ── Step 2: Find nearest expiry >= reference_date ───────────────────────
    target_date = datetime.strptime(reference_date, '%Y-%m-%d') if reference_date else datetime.now()
    target_expiry_str = None

    for exp in sorted(expiries):          # expiries should already be sorted ASC
        exp_dt = datetime.strptime(str(exp), '%Y-%m-%d')
        if exp_dt.date() >= target_date.date():
            target_expiry_str = str(exp)
            break

    if not target_expiry_str:
        # All expiries are before the target date — use the most recent one
        target_expiry_str = str(sorted(expiries)[-1])
        print(f"[ExpiredFetcher] No expiry >= {target_date.date()}. "
              f"Using most recent available: {target_expiry_str}")

    target_expiry_dt = datetime.strptime(target_expiry_str, '%Y-%m-%d')
    print(f"[ExpiredFetcher] Target expiry resolved to: {target_expiry_str}")

    # ── Step 3: Fetch all contracts for the selected expiry ──────────────────
    contracts = fetcher.fetch_contracts(underlying, target_expiry_str)
    if not contracts:
        print(f"[ExpiredFetcher] No contracts found for expiry {target_expiry_str}.")
        return [], target_expiry_dt

    df = pd.DataFrame([c.to_dict() if hasattr(c, 'to_dict') else c for c in contracts])
    print(f"[ExpiredFetcher] Total contracts fetched: {len(df)}")

    # Normalise column names from the expired API response
    df['strike'] = df['strike_price'].astype(float)
    df['type']   = df['instrument_type']           # 'CE' or 'PE'

    # ── Step 4: Select ATM ± num_strikes around spot_price ──────────────────
    strikes = sorted(df['strike'].unique())
    atm_idx = int(np.abs(np.array(strikes) - spot_price).argmin())
    atm_strike = strikes[atm_idx]
    print(f"[ExpiredFetcher] ATM strike: {atm_strike} (spot={spot_price})")

    lo = max(0, atm_idx - num_strikes)
    hi = min(len(strikes), atm_idx + num_strikes + 1)
    selected_strikes = strikes[lo:hi]

    final_instruments = []
    for strike in selected_strikes:
        try:
            ce_rows = df[(df['strike'] == strike) & (df['type'] == 'CE')]
            pe_rows = df[(df['strike'] == strike) & (df['type'] == 'PE')]

            if not ce_rows.empty:
                ce = ce_rows.iloc[0]
                final_instruments.append({
                    'key':    ce['instrument_key'],   # e.g. "NSE_FO|64359|24-02-2026"
                    'symbol': ce['trading_symbol'],
                    'strike': strike,
                    'type':   'CE',
                    'expiry': target_expiry_dt
                })
            if not pe_rows.empty:
                pe = pe_rows.iloc[0]
                final_instruments.append({
                    'key':    pe['instrument_key'],
                    'symbol': pe['trading_symbol'],
                    'strike': strike,
                    'type':   'PE',
                    'expiry': target_expiry_dt
                })
        except Exception as e:
            print(f"[ExpiredFetcher] Error processing strike {strike}: {e}")
            continue

    print(f"[ExpiredFetcher] Selected {len(final_instruments)} instruments around ATM {atm_strike}")
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
    """
    Strategy for fetching data for Expired option contracts.

    Flow:
      1. get_spot_price  → HistoricalCandleFetcher fetches 5-min index candles for the
                          selected date and returns the close price at target_time_str.
                          This price is the ATM reference.
      2. get_instruments → ExpiredCandleFetcher resolves expiry list, finds nearest
                          expiry >= selected date, downloads all contracts, and selects
                          strikes around the ATM price from step 1.
      3. get_candle_data → Fetches single-day expired historical candles via the
                          Upstox expired-instruments URL:
                          /historical-candle/<ins_key>/5minute/<date>/<date>
    """

    def __init__(self, symbol="NIFTY"):
        self.symbol       = symbol.upper()
        self.fetcher      = ExpiredCandleFetcher()
        self.hist_fetcher = HistoricalCandleFetcher()
        # Spot/index key — indices are NOT available via the Expired API,
        # so we always use HistoricalCandleFetcher for the spot.
        self.spot_key = INDEX_KEYS.get(self.symbol, INDEX_KEYS["NIFTY"])

    # ── Step 1: Spot price at selected time (ATM reference) ──────────────────
    def get_spot_price(self, target_date_str, target_time_str):
        """
        Returns the index close price at the nearest 5-min candle to target_time_str.
        Falls back to the daily close if intraday data is unavailable.
        """
        spot_price = None
        target_dt  = datetime.strptime(target_date_str, '%Y-%m-%d')

        # ── Intraday path (time selected) ───────────────────────────────────
        if target_time_str:
            print(f"[ExpiredStrategy] Fetching 5-min {self.symbol} spot for "
                  f"{target_date_str} to find ATM at {target_time_str}...")
            spot_df = self.hist_fetcher._fetch_single(
                self.spot_key, "minutes", 5,
                target_date_str, target_date_str
            )

            if spot_df is not None and not spot_df.empty:
                target_full_dt = datetime.strptime(
                    f"{target_date_str} {target_time_str}", "%Y-%m-%d %H:%M"
                )
                try:
                    # get_indexer with method='nearest' works even for tz-aware indexes
                    nearest_idx = spot_df.index.get_indexer(
                        [target_full_dt], method='nearest'
                    )[0]
                    spot_price = float(spot_df.iloc[nearest_idx]['close'])
                    print(f"[ExpiredStrategy] ATM price at "
                          f"{spot_df.index[nearest_idx]}: {spot_price}")
                except Exception as e:
                    print(f"[ExpiredStrategy] get_indexer error: {e}")

        # ── Daily fallback ───────────────────────────────────────────────────
        if spot_price is None:
            print(f"[ExpiredStrategy] Falling back to daily close for {target_date_str}...")
            spot_df_daily = self.hist_fetcher.fetch(
                self.spot_key, timeframe="days", lookback_days=30
            )

            if spot_df_daily is not None and not spot_df_daily.empty:
                available = spot_df_daily.index.strftime('%Y-%m-%d')
                if target_date_str in available:
                    val = spot_df_daily.loc[target_date_str]['close']
                    spot_price = float(val.iloc[0] if isinstance(val, pd.Series) else val)
                    print(f"[ExpiredStrategy] Daily close on {target_date_str}: {spot_price}")
                else:
                    # Nearest prior trading day
                    td_aware = target_dt
                    if spot_df_daily.index.tzinfo is not None:
                        td_aware = target_dt.replace(tzinfo=spot_df_daily.index.tzinfo)
                    past = spot_df_daily[spot_df_daily.index <= td_aware]
                    if not past.empty:
                        spot_price = float(past['close'].iloc[-1])
                        print(f"[ExpiredStrategy] Using nearest prior close "
                              f"({past.index[-1].date()}): {spot_price}")

        return spot_price

    # ── Step 2: Build instrument list around ATM ─────────────────────────────
    def get_instruments(self, spot_price, target_date_str):
        """
        Resolves expiry, downloads contracts, and returns strikes near ATM.
        The spot_price here is already the price at the selected time (from step 1).
        """
        instruments, expiry = get_expired_option_chain_instruments(
            spot_price,
            num_strikes=3,
            reference_date=target_date_str,
            symbol=self.symbol
        )
        return instruments, expiry, True   # is_expired=True

    # ── Step 3: IV spot map (full day, 5-min) ────────────────────────────────
    def get_iv_spot_data(self, target_date_str):
        """Fetch full-day 5-min spot candles for IV calculation map."""
        return self.hist_fetcher._fetch_single(
            self.spot_key, "minutes", 5,
            target_date_str, target_date_str
        )

    # ── Step 4: Expired candle data for a single option instrument ───────────
    def get_candle_data(self, instrument, from_date, to_date):
        """
        Fetch 5-min candles for an expired option on the target date using:
        GET /v2/expired-instruments/historical-candle/<ins_key>/5minute/<to_date>/<to_date>

        The expired API instrument_key already contains the expiry suffix
        (e.g. NSE_FO|64844|24-02-2026), so no extra formatting is needed.
        We always request just the single day (to_date) because the contract
        only trades around its expiry window.
        """
        instr_key = instrument.get('key', '')

        # Safety: append expiry suffix if key is missing it (legacy format)
        expiry_dt = instrument.get('expiry')
        if instr_key.count('|') < 2 and expiry_dt:
            instr_key = f"{instr_key}|{expiry_dt.strftime('%d-%m-%Y')}"

        print(f"[ExpiredStrategy] Fetching expired candles: key={instr_key}, date={to_date}")

        # Single day: from_date == to_date == target_date_str
        return self.fetcher.fetch_candle_data(instr_key, "5minute", to_date, to_date)

class OptionChainProcessor:
    """Context class that executes the data processing workflow using a Strategy."""
    
    def __init__(self, strategy: MarketDataStrategy):
        self.strategy = strategy
        
    def process_data(self, df, spot_map, instr, filter_date=None):
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
        
        # If a specific date was requested, filter to that date ONLY to avoid slanted lines across multiple days
        if filter_date:
            final_df = final_df[final_df['date'].dt.strftime('%Y-%m-%d') == filter_date]
        
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
                from_date = target_date_str
                
                df_candles = self.strategy.get_candle_data(inst, from_date, to_date)
                if df_candles is not None and not df_candles.empty:
                    processed = self.process_data(df_candles, spot_map, inst, filter_date=target_date_str)
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
            if os.path.exists(filename):
                os.remove(filename)
            
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
            if os.path.exists(filename):
                os.remove(filename)
            
        with open(filename, 'w') as f:
            json.dump(meta, f, indent=4)
        print(f"Saved metadata to {filename}")



def _fetch_last_expired_date(symbol: str) -> datetime | None:
    """
    Query the Upstox expired-instruments/expiries endpoint and return
    the most recent (highest) expiry date as a datetime, or None on failure.
    """
    try:
        underlying = INDEX_KEYS.get(symbol, INDEX_KEYS["NIFTY"])
        fetcher = ExpiredCandleFetcher()
        expiries = fetcher.fetch_expiries(underlying)
        if expiries:
            return datetime.strptime(str(sorted(expiries)[-1]), '%Y-%m-%d')
    except Exception as e:
        print(f"[StrategyDetect] Could not fetch expired expiries: {e}")
    return None


def _detect_strategy(target_date_str: str,
                     target_time_str: str | None,
                     symbol: str,
                     live_mode: bool) -> MarketDataStrategy:
    """
    Determines the correct fetching strategy based on the selected date/time.

    Three mutually-exclusive zones
    ───────────────────────────────────────────────────────────────────────────
    Zone 1 │ LIVE  (explicit --live flag only)
    ───────┤ Condition : live_mode == True
           │ Strategy  : LiveStrategy — real-time intraday candles, no time reference
    ───────────────────────────────────────────────────────────────────────────
    Zone 2 │ EXPIRED
    ───────┤ Condition : selected_date <= most_recent_date in expired-API list
           │ Strategy  : ExpiredStrategy  — uses /v2/expired-instruments/* APIs
    ───────────────────────────────────────────────────────────────────────────
    Zone 3 │ HISTORICAL (active contract window)
    ───────┤ Condition : selected_date > last_expired_date
           │             AND selected_date <= today  (or today with a time_str)
           │ Strategy  : HistoricalStrategy  — uses /v2/historical-candle/* API
    ───────────────────────────────────────────────────────────────────────────
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_dt  = datetime.strptime(today_str, '%Y-%m-%d')
    target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')

    # ── Zone 1: Explicit --live flag ONLY ──────────────────────────────────
    #   Today's date is reserved strictly for Live mode. If the --live flag is
    #   missing, any request for today's date will be blocked by the guard below.
    if live_mode:
        print(f"[StrategyDetect] Zone 1 -> LIVE  (live_flag=True, date={target_date_str})")
        return LiveStrategy()

    # ── Guard: today's date without live mode ────────────────────────────────
    #   Today is reserved for Live mode.
    if target_date_str == today_str:
        print(f"[StrategyDetect] BLOCKED: {target_date_str} is today. "
              f"Use --live flag or the Live toggle for today's data.")
        # Return HistoricalStrategy as a safe fallback that will find no data
        return HistoricalStrategy()


    # ── Boundary: fetch most recent expired-contract date ───────────────────
    last_expired_dt = _fetch_last_expired_date(symbol)

    print(f"[StrategyDetect] Boundaries — "
          f"selected={target_date_str}  "
          f"last_expired={last_expired_dt.date() if last_expired_dt else 'N/A'}  "
          f"today={today_str}")

    # ── Zone 2: Expired ─────────────────────────────────────────────────────
    #   selected_date <= last_expired_date  →  contract has already expired;
    #   data is only available via the /v2/expired-instruments/ API.
    if last_expired_dt and target_dt <= last_expired_dt:
        print(f"[StrategyDetect] Zone 2 -> EXPIRED  "
              f"({target_date_str} <= last_expired={last_expired_dt.date()})")
        return ExpiredStrategy(symbol=symbol)

    # ── Zone 3: Historical ──────────────────────────────────────────────────
    #   selected_date is AFTER the last expired contract but <= today.
    #   The contract is still active (not yet in the expired list) so
    #   data lives in the standard /v2/historical-candle/ API.
    #   Also covers: today with a time_str (intraday-historical slice).
    if target_dt <= today_dt:
        if target_time_str:
            print(f"[StrategyDetect] Zone 3 -> HISTORICAL (intraday-slice)  "
                  f"({target_date_str} {target_time_str})")
        else:
            print(f"[StrategyDetect] Zone 3 -> HISTORICAL  "
                  f"({last_expired_dt.date() if last_expired_dt else 'N/A'} "
                  f"< {target_date_str} <= {today_str})")
        return HistoricalStrategy()

    # ── Future date (no data) ───────────────────────────────────────────────
    print(f"[StrategyDetect] WARNING: {target_date_str} is in the FUTURE. "
          f"Defaulting to HistoricalStrategy (will likely return no data).")
    return HistoricalStrategy()


def main():
    live_mode = "--live" in sys.argv

    # Optional --symbol=NIFTY / --symbol=BANKNIFTY flag
    symbol = "NIFTY"
    for arg in sys.argv[1:]:
        if arg.startswith("--symbol="):
            symbol = arg.split("=", 1)[1].upper()
            break

    # Positional args: date (index 0) and optional time (index 1)
    today_str  = datetime.now().strftime('%Y-%m-%d')
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]

    target_date_str = positional[0] if positional else today_str
    target_time_str = positional[1] if len(positional) > 1 else None

    strategy = _detect_strategy(target_date_str, target_time_str, symbol, live_mode)

    processor = OptionChainProcessor(strategy)
    processor.run(target_date_str, target_time_str)


if __name__ == "__main__":
    main()
