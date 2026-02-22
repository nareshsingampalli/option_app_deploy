import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ============================================================
# VOLATILITY SQUEEZE INDICATORS
# ============================================================

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands and Band Width."""
    close = df['close']
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Bollinger Band Width (BBW) - key squeeze indicator
    bbw = (upper_band - lower_band) / sma * 100
    
    return {
        'upper': upper_band,
        'lower': lower_band,
        'middle': sma,
        'bbw': bbw,
        'std': std
    }

def calculate_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5):
    """Calculate Keltner Channels using ATR."""
    close = df['close']
    high = df['high']
    low = df['low']
    
    # EMA for middle line
    ema = close.ewm(span=period, adjust=False).mean()
    
    # ATR calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    upper_kc = ema + (atr * atr_mult)
    lower_kc = ema - (atr * atr_mult)
    
    return {
        'upper': upper_kc,
        'lower': lower_kc,
        'middle': ema,
        'atr': atr
    }

def detect_ttm_squeeze(df: pd.DataFrame) -> dict:
    """
    TTM Squeeze Detection (John Carter's method).
    Squeeze = Bollinger Bands INSIDE Keltner Channels.
    This indicates extremely low volatility.
    """
    if len(df) < 20:
        return {"squeeze_on": False, "squeeze_strength": 0, "bars_in_squeeze": 0}
    
    bb = calculate_bollinger_bands(df)
    kc = calculate_keltner_channels(df)
    
    # Squeeze is ON when BB is inside KC
    squeeze_on = (bb['lower'].iloc[-1] > kc['lower'].iloc[-1]) and \
                 (bb['upper'].iloc[-1] < kc['upper'].iloc[-1])
    
    # Count consecutive bars in squeeze
    bars_in_squeeze = 0
    for i in range(len(df) - 1, -1, -1):
        if pd.isna(bb['lower'].iloc[i]) or pd.isna(kc['lower'].iloc[i]):
            break
        if (bb['lower'].iloc[i] > kc['lower'].iloc[i]) and \
           (bb['upper'].iloc[i] < kc['upper'].iloc[i]):
            bars_in_squeeze += 1
        else:
            break
    
    # Squeeze strength: how tight are the bands?
    if not pd.isna(bb['bbw'].iloc[-1]) and not pd.isna(bb['bbw'].rolling(100).min().iloc[-1]):
        bbw_current = bb['bbw'].iloc[-1]
        bbw_min = bb['bbw'].rolling(min(100, len(df))).min().iloc[-1]
        bbw_max = bb['bbw'].rolling(min(100, len(df))).max().iloc[-1]
        if bbw_max > bbw_min:
            squeeze_strength = 1 - (bbw_current - bbw_min) / (bbw_max - bbw_min)
        else:
            squeeze_strength = 0
    else:
        squeeze_strength = 0
    
    return {
        "squeeze_on": squeeze_on,
        "squeeze_strength": round(squeeze_strength, 3),
        "bars_in_squeeze": bars_in_squeeze,
        "bbw_current": round(bb['bbw'].iloc[-1], 3) if not pd.isna(bb['bbw'].iloc[-1]) else 0
    }

def detect_atr_compression(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> dict:
    """
    ATR-based volatility compression.
    Compares short-term ATR to long-term ATR.
    """
    if len(df) < long_period:
        return {"atr_ratio": 1.0, "compressed": False, "compression_pct": 0}
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_short = tr.rolling(window=short_period).mean().iloc[-1]
    atr_long = tr.rolling(window=long_period).mean().iloc[-1]
    
    atr_ratio = atr_short / atr_long if atr_long > 0 else 1.0
    compressed = atr_ratio < 0.7  # Short ATR is 30% lower than long ATR
    compression_pct = max(0, (1 - atr_ratio)) * 100
    
    return {
        "atr_ratio": round(atr_ratio, 3),
        "compressed": compressed,
        "compression_pct": round(compression_pct, 1),
        "atr_short": round(atr_short, 2) if not pd.isna(atr_short) else 0,
        "atr_long": round(atr_long, 2) if not pd.isna(atr_long) else 0
    }

def detect_bbw_squeeze(df: pd.DataFrame, lookback: int = 100) -> dict:
    """
    Bollinger Band Width Squeeze Detection.
    Identifies when BBW is at historical lows.
    """
    if len(df) < 20:
        return {"squeeze_percentile": 100, "is_squeeze": False}
    
    bb = calculate_bollinger_bands(df)
    bbw = bb['bbw'].dropna()
    
    if len(bbw) < 10:
        return {"squeeze_percentile": 100, "is_squeeze": False}
    
    current_bbw = bbw.iloc[-1]
    lookback_period = min(lookback, len(bbw))
    historical_bbw = bbw.iloc[-lookback_period:]
    
    # Calculate percentile rank
    percentile = (historical_bbw < current_bbw).sum() / len(historical_bbw) * 100
    
    return {
        "squeeze_percentile": round(percentile, 1),
        "is_squeeze": percentile < 20,  # Bottom 20% = squeeze
        "current_bbw": round(current_bbw, 3),
        "min_bbw": round(historical_bbw.min(), 3),
        "max_bbw": round(historical_bbw.max(), 3)
    }

# ============================================================
# IMPROVED RANGE SHRINKAGE
# ============================================================

def test_range_shrinkage_improved(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Improved Range Shrinkage with rolling comparison.
    Compares recent range to historical range.
    """
    if len(df) < lookback * 2:
        # Fallback to simple calculation
        highest_high = df["high"].max()
        lowest_low = df["low"].min()
        mid_price = (highest_high + lowest_low) / 2
        range_percent = ((highest_high - lowest_low) / mid_price) * 100 if mid_price > 0 else 100
        return {
            "range_percent": round(range_percent, 2),
            "range_ratio": 1.0,
            "is_compressed": range_percent < 5,
            "method": "simple"
        }
    
    # Recent range (last N bars)
    recent = df.iloc[-lookback:]
    recent_range = recent['high'].max() - recent['low'].min()
    
    # Historical range (previous N bars)
    historical = df.iloc[-lookback*2:-lookback]
    historical_range = historical['high'].max() - historical['low'].min()
    
    # Range ratio
    range_ratio = recent_range / historical_range if historical_range > 0 else 1.0
    
    # Current range as percentage
    mid_price = (df['high'].max() + df['low'].min()) / 2
    range_percent = ((recent_range) / mid_price) * 100 if mid_price > 0 else 100
    
    return {
        "range_percent": round(range_percent, 2),
        "range_ratio": round(range_ratio, 3),
        "is_compressed": range_ratio < 0.5 or range_percent < 5,  # 50% compression OR < 5%
        "recent_range": round(recent_range, 2),
        "historical_range": round(historical_range, 2),
        "method": "rolling_comparison"
    }

# ============================================================
# IMPROVED SLOPE DETECTION
# ============================================================

def test_slope_neutrality_improved(df: pd.DataFrame, lookback: int = None) -> dict:
    """
    Improved Slope Neutrality with R² confidence and normalized thresholds.
    """
    if lookback is None or lookback > len(df):
        recent_df = df.copy()
    else:
        recent_df = df.iloc[-lookback:].copy()
    
    if len(recent_df) < 10:
        return {
            "high_slope": {"value": 0, "normalized": 0, "r2": 0, "passed": False},
            "low_slope": {"value": 0, "normalized": 0, "r2": 0, "passed": False}
        }
    
    # Get swing points
    window = max(3, len(recent_df) // 10)
    rolling_max = recent_df['high'].rolling(window=window*2+1, center=True).max()
    rolling_min = recent_df['low'].rolling(window=window*2+1, center=True).min()
    
    swing_highs = recent_df[recent_df['high'] == rolling_max].dropna()
    swing_lows = recent_df[recent_df['low'] == rolling_min].dropna()
    
    results = {}
    price_range = recent_df['high'].max() - recent_df['low'].min()
    
    # High slope analysis
    if len(swing_highs) >= 2:
        X = np.arange(len(swing_highs)).reshape(-1, 1)
        y = swing_highs['high'].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        r2 = model.score(X, y)
        
        # Normalize slope by price range and number of points
        normalized_slope = abs(slope) / price_range * len(swing_highs) if price_range > 0 else 0
        
        results["high_slope"] = {
            "value": round(slope, 6),
            "normalized": round(normalized_slope, 4),
            "r2": round(r2, 3),
            "passed": normalized_slope < 0.15 and r2 > 0.3,  # Low slope with decent fit
            "num_points": len(swing_highs)
        }
    else:
        results["high_slope"] = {"value": 0, "normalized": 0, "r2": 0, "passed": False, "num_points": 0}
    
    # Low slope analysis
    if len(swing_lows) >= 2:
        X = np.arange(len(swing_lows)).reshape(-1, 1)
        y = swing_lows['low'].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        r2 = model.score(X, y)
        
        normalized_slope = abs(slope) / price_range * len(swing_lows) if price_range > 0 else 0
        
        results["low_slope"] = {
            "value": round(slope, 6),
            "normalized": round(normalized_slope, 4),
            "r2": round(r2, 3),
            "passed": normalized_slope < 0.15 and r2 > 0.3,
            "num_points": len(swing_lows)
        }
    else:
        results["low_slope"] = {"value": 0, "normalized": 0, "r2": 0, "passed": False, "num_points": 0}
    
    return results

# ============================================================
# ORIGINAL FUNCTIONS (kept for backward compatibility)
# ============================================================

def get_swing_highs_lows(df: pd.DataFrame, window: int = 10):
    """
    Finds swing highs and lows in the dataframe.
    A swing high is a peak higher than the prices around it.
    A swing low is a trough lower than the prices around it.
    """
    # Find rolling max/min
    rolling_max = df['high'].rolling(window=window*2+1, center=True).max()
    rolling_min = df['low'].rolling(window=window*2+1, center=True).min()

    # A swing high is a point where the high is the rolling max
    swing_highs = df[df['high'] == rolling_max]
    
    # A swing low is a point where the low is the rolling min
    swing_lows = df[df['low'] == rolling_min]

    return swing_highs, swing_lows

def test_range_percentage_shrinkage(df: pd.DataFrame, lookback: int = None, threshold: float = 8.0) -> bool:
    """
    Test 1: Range Percentage Shrinkage
    Checks if the price range has compressed below a certain percentage.
    """
    # Use all available data if lookback is not specified or too large
    if lookback is None or lookback > len(df):
        recent_df = df
    else:
        recent_df = df.iloc[-lookback:]

    highest_high = recent_df["high"].max()
    lowest_low = recent_df["low"].min()
    mid_price = (highest_high + lowest_low) / 2

    if mid_price == 0:
        return False

    range_percent = ((highest_high - lowest_low) / mid_price) * 100
    return range_percent <= threshold

def test_slope_neutrality(df: pd.DataFrame, lookback: int = None, threshold: float = 0.01) -> dict:
    """
    Test 2: Slope Neutrality
    Checks if the slopes of swing highs and lows are close to zero for a rectangle.
    Returns individual results for high and low slope neutrality.
    """
    # Use all available data if lookback is not specified or too large
    if lookback is None or lookback > len(df):
        recent_df = df.copy()
    else:
        recent_df = df.iloc[-lookback:].copy()
    
    swing_highs, swing_lows = get_swing_highs_lows(recent_df)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"high_slope_neutrality": False, "low_slope_neutrality": False}

    # Highs - use numeric indices for LinearRegression
    X_high = np.arange(len(swing_highs)).reshape(-1, 1)
    y_high = swing_highs['high'].values
    model_high = LinearRegression()
    model_high.fit(X_high, y_high)
    high_slope = model_high.coef_[0]

    # Lows - use numeric indices for LinearRegression
    X_low = np.arange(len(swing_lows)).reshape(-1, 1)
    y_low = swing_lows['low'].values
    model_low = LinearRegression()
    model_low.fit(X_low, y_low)
    low_slope = model_low.coef_[0]

    return {
        "high_slope_neutrality": abs(high_slope) < threshold,
        "low_slope_neutrality": abs(low_slope) < threshold
    }

def test_range_stability(df: pd.DataFrame, lookback: int = None, threshold_ratio: float = 0.1) -> bool:
    """
    Test 3: Range Stability
    Checks if the standard deviation of highs and lows is small relative to the range.
    """
    # Use all available data if lookback is not specified or too large
    if lookback is None or lookback > len(df):
        recent_df = df
    else:
        recent_df = df.iloc[-lookback:]
    std_high = recent_df["high"].std()
    std_low = recent_df["low"].std()

    price_range = recent_df["high"].max() - recent_df["low"].min()
    
    if price_range == 0:
        return True # If range is zero, it's stable

    # Threshold is a fraction of the price range
    threshold = price_range * threshold_ratio
    
    return std_high < threshold and std_low < threshold

def test_candle_overlap_ratio(df: pd.DataFrame, lookback: int = None, threshold: float = 0.6) -> bool:
    """
    Test 4: Candle Overlap Ratio
    Checks if a high percentage of recent candles overlap with each other.
    """
    # Use all available data if lookback is not specified or too large
    if lookback is None or lookback > len(df):
        recent_df = df
    else:
        recent_df = df.iloc[-lookback:]
    overlapping_candles = 0

    for i in range(1, len(recent_df)):
        high_t = recent_df["high"].iloc[i]
        low_t = recent_df["low"].iloc[i]
        high_t_minus_1 = recent_df["high"].iloc[i - 1]
        low_t_minus_1 = recent_df["low"].iloc[i - 1]

        overlap = min(high_t, high_t_minus_1) - max(low_t, low_t_minus_1)
        if overlap > 0:
            overlapping_candles += 1

    overlap_ratio = overlapping_candles / (len(recent_df) - 1)
    return overlap_ratio >= threshold

def test_body_compression(df: pd.DataFrame, short_lookback: int = None, long_lookback: int = None, threshold: float = 0.7) -> bool:
    """
    Test 5: Body Compression
    Checks if the average candle body size has recently decreased.
    """
    # Set default lookbacks if not provided
    if short_lookback is None:
        short_lookback = min(10, len(df) // 3)
    if long_lookback is None:
        long_lookback = min(30, len(df) // 2)
    
    # Ensure we have enough data
    if len(df) < long_lookback + short_lookback:
        return False

    body_size = (df["close"] - df["open"]).abs()
    
    # Ensure we have enough data for the intended slices
    if len(body_size) < long_lookback + short_lookback:
        return False

    avg_body_short = body_size.iloc[-short_lookback:].mean()
    # Ensure the 'long' period doesn't overlap with the 'short' one
    avg_body_long = body_size.iloc[-(long_lookback + short_lookback):-short_lookback].mean()
    
    if avg_body_long == 0:
        # If the long-term average is zero, we can't show a reduction.
        # This could be a doji-filled period. If short-term is also zero, it's compressed.
        return avg_body_short == 0

    return avg_body_short < (avg_body_long * threshold)

def detect_type2_compression(df: pd.DataFrame) -> dict:
    """
    Runs all compression tests and returns a score and detailed results with actual values.
    Updated to use < 5% range compression and entry/exit timing.
    """
    # Ensure dataframe has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

    # Sort by date
    df = df.sort_index()
    
    # Initialize results dictionary
    results = {}
    
    # --- Liquidity Asymmetry Score ---
    # If bid/ask volume is available, use it. Otherwise, use up/down volume as a proxy.
    if 'volume' in df.columns:
        up_volume = df[df['close'] > df['open']]['volume'].sum()
        down_volume = df[df['close'] < df['open']]['volume'].sum()
        if up_volume + down_volume > 0:
            liquidity_asymmetry_score = (up_volume - down_volume) / (up_volume + down_volume)
        else:
            liquidity_asymmetry_score = 0
        results["liquidity_asymmetry_score"] = round(liquidity_asymmetry_score, 3)
    else:
        results["liquidity_asymmetry_score"] = 0
    
    # --- Compression Location (Context) ---
    last_close = df['close'].iloc[-1]
    highest_high = df['high'].max()
    lowest_low = df['low'].min()
    range_width = highest_high - lowest_low
    location = "middle"
    if range_width > 0:
        rel_pos = (last_close - lowest_low) / range_width
        if rel_pos > 0.8:
            location = "at highs"
        elif rel_pos < 0.2:
            location = "at lows"
    
    # Institutional interpretation
    if location == "at highs":
        institutional = "Distribution / bull trap risk"
    elif location == "at lows":
        institutional = "Accumulation / short squeeze risk"
    else:
        institutional = "Neutral / range-bound"
    results["compression_location"] = {
        "context": location,
        "institutional_interpretation": institutional
    }
    
    # --- Compression Duration Score ---
    # Note: Simplified to avoid recursion - just set to 0 for now
    # Full duration calculation would require a separate helper function
    results["compression_duration_score"] = 0
    
    # Test 1: Range Percentage with actual value (updated to < 5%)
    highest_high = df["high"].max()
    lowest_low = df["low"].min()
    mid_price = (highest_high + lowest_low) / 2
    range_percent = ((highest_high - lowest_low) / mid_price) * 100
    results["range_shrinkage"] = {
        "value": round(range_percent, 2),
        "threshold": 5.0,  # Updated to 5%
        "passed": range_percent < 5.0  # Updated to < 5%
    }
    
    # Test 2: Slope Neutrality with actual values
    swing_highs, swing_lows = get_swing_highs_lows(df)
    slope_results = {}
    
    if len(swing_highs) >= 2:
        X_high = np.arange(len(swing_highs)).reshape(-1, 1)
        y_high = swing_highs['high'].values
        model_high = LinearRegression()
        model_high.fit(X_high, y_high)
        high_slope = model_high.coef_[0]
        slope_results["high_slope_neutrality"] = {
            "value": round(high_slope, 6),
            "threshold": 0.01,
            "passed": abs(high_slope) < 0.01
        }
    else:
        slope_results["high_slope_neutrality"] = {
            "value": 0,
            "threshold": 0.01,
            "passed": False
        }
    
    if len(swing_lows) >= 2:
        X_low = np.arange(len(swing_lows)).reshape(-1, 1)
        y_low = swing_lows['low'].values
        model_low = LinearRegression()
        model_low.fit(X_low, y_low)
        low_slope = model_low.coef_[0]
        slope_results["low_slope_neutrality"] = {
            "value": round(low_slope, 6),
            "threshold": 0.01,
            "passed": abs(low_slope) < 0.01
        }
    else:
        slope_results["low_slope_neutrality"] = {
            "value": 0,
            "threshold": 0.01,
            "passed": False
        }
    
    results.update(slope_results)
    
    # Test 3: Range Stability with actual values
    std_high = df["high"].std()
    std_low = df["low"].std()
    price_range = df["high"].max() - df["low"].min()
    threshold_stability = price_range * 0.1
    results["range_stability"] = {
        "high_std": round(std_high, 2),
        "low_std": round(std_low, 2),
        "threshold": round(threshold_stability, 2),
        "passed": std_high < threshold_stability and std_low < threshold_stability
    }
    
    # Test 4: Candle Overlap Ratio with actual value
    overlapping_candles = 0
    for i in range(1, len(df)):
        high_t = df["high"].iloc[i]
        low_t = df["low"].iloc[i]
        high_t_minus_1 = df["high"].iloc[i - 1]
        low_t_minus_1 = df["low"].iloc[i - 1]
        overlap = min(high_t, high_t_minus_1) - max(low_t, low_t_minus_1)
        if overlap > 0:
            overlapping_candles += 1
    
    overlap_ratio = overlapping_candles / (len(df) - 1) if len(df) > 1 else 0
    results["candle_overlap"] = {
        "value": round(overlap_ratio * 100, 1),
        "threshold": 65.0,
        "passed": overlap_ratio >= 0.65
    }
    
    # Test 5: Body Compression with actual values
    body_size = (df["close"] - df["open"]).abs()
    short_lookback = min(10, len(df) // 3)
    long_lookback = min(30, len(df) // 2)
    
    if len(body_size) >= long_lookback + short_lookback:
        avg_body_short = body_size.iloc[-short_lookback:].mean()
        avg_body_long = body_size.iloc[-(long_lookback + short_lookback):-short_lookback].mean()
        
        if avg_body_long > 0:
            compression_ratio = avg_body_short / avg_body_long
            results["body_compression"] = {
                "value": round(compression_ratio, 3),
                "threshold": 0.7,
                "passed": compression_ratio < 0.7
            }
        else:
            results["body_compression"] = {
                "value": 0,
                "threshold": 0.7,
                "passed": False
            }
    else:
        results["body_compression"] = {
            "value": 0,
            "threshold": 0.7,
            "passed": False
        }

    # ============================================================
    # VOLATILITY SQUEEZE INDICATORS (New!)
    # ============================================================
    
    # TTM Squeeze (Bollinger inside Keltner)
    ttm_result = detect_ttm_squeeze(df)
    results["ttm_squeeze"] = ttm_result
    
    # ATR Compression
    atr_result = detect_atr_compression(df)
    results["atr_compression"] = atr_result
    
    # BBW Squeeze
    bbw_result = detect_bbw_squeeze(df)
    results["bbw_squeeze"] = bbw_result
    
    # Improved Range Shrinkage
    range_improved = test_range_shrinkage_improved(df)
    results["range_shrinkage_improved"] = range_improved
    
    # Improved Slope Analysis
    slope_improved = test_slope_neutrality_improved(df)
    results["slope_analysis_improved"] = slope_improved

    # ============================================================
    # WEIGHTED SCORING SYSTEM (IMPROVED)
    # ============================================================
    # Core tests: 80% (8 points max) - 5 components x 1.6 points
    # Volatility Squeeze Bonus: 10% (1 point max)
    # Pattern Bonus: 10% (1 point max)
    
    core_score = 0.0
    
    # 1. Range Shrinkage (Improved): 1.6 points
    # Uses rolling comparison + absolute threshold
    range_res = results["range_shrinkage_improved"]
    range_contribution = 0.0
    
    if range_res["is_compressed"]:
        range_contribution = 1.6
    else:
        # Partial credit:
        # If range_percent < 8% (close to 5%)
        if range_res["range_percent"] < 8.0:
            range_contribution += 0.8 * (1 - (range_res["range_percent"] - 5.0) / 3.0)
        # If range_ratio < 0.8 (close to 0.5)
        if range_res["range_ratio"] < 0.8:
            range_contribution += 0.8 * (1 - (range_res["range_ratio"] - 0.5) / 0.3)
            
    range_contribution = min(1.6, range_contribution)
    core_score += range_contribution
    results["range_shrinkage"]["score_contribution"] = round(range_contribution, 2)
    
    # 2. High Slope Neutrality (Improved): 1.6 points
    high_slope_res = results["slope_analysis_improved"]["high_slope"]
    high_slope_contribution = 0.0
    
    if high_slope_res["passed"]:
        high_slope_contribution = 1.6
    else:
        # Partial credit based on normalized slope
        norm_slope = high_slope_res["normalized"]
        if norm_slope < 0.3:
            high_slope_contribution = 1.6 * (1 - norm_slope / 0.3)
    
    core_score += high_slope_contribution
    results["high_slope_neutrality"]["score_contribution"] = round(high_slope_contribution, 2)
    
    # 3. Low Slope Neutrality (Improved): 1.6 points
    low_slope_res = results["slope_analysis_improved"]["low_slope"]
    low_slope_contribution = 0.0
    
    if low_slope_res["passed"]:
        low_slope_contribution = 1.6
    else:
        # Partial credit based on normalized slope
        norm_slope = low_slope_res["normalized"]
        if norm_slope < 0.3:
            low_slope_contribution = 1.6 * (1 - norm_slope / 0.3)
            
    core_score += low_slope_contribution
    results["low_slope_neutrality"]["score_contribution"] = round(low_slope_contribution, 2)
    
    # 4. Candle Overlap: 1.6 points
    overlap_contribution = 0.0
    overlap_val = results["candle_overlap"]["value"] / 100
    if overlap_val >= 0.65:
        overlap_contribution = 1.6
    elif overlap_val >= 0.5:
        overlap_contribution = 1.6 * ((overlap_val - 0.5) / 0.15)
        
    core_score += overlap_contribution
    results["candle_overlap"]["score_contribution"] = round(overlap_contribution, 2)
    
    # 5. Body Compression: 1.6 points
    body_contribution = 0.0
    if results["body_compression"].get("passed", False):
        body_contribution = 1.6
    else:
        body_val = results["body_compression"].get("value", 1)
        if body_val < 0.9 and body_val > 0:
            body_contribution = 1.6 * (1 - (body_val - 0.7) / 0.2)
            body_contribution = max(0, body_contribution)
            
    core_score += body_contribution
    results["body_compression"]["score_contribution"] = round(body_contribution, 2)
    
    # Volatility Squeeze Bonus: 10% (1 point max)
    volatility_bonus = 0.0
    
    # TTM Squeeze is strongest indicator
    if results["ttm_squeeze"]["squeeze_on"]:
        volatility_bonus = 1.0
    else:
        # Partial bonus from ATR compression or BBW squeeze
        if results["atr_compression"]["compressed"]:
            volatility_bonus = max(volatility_bonus, 0.5 + results["atr_compression"]["compression_pct"]/200)
        
        if results["bbw_squeeze"]["is_squeeze"]:
            volatility_bonus = max(volatility_bonus, 0.5)
            
    volatility_bonus = min(1.0, volatility_bonus)
    
    # Pattern Detection Bonus: 10% (1 point max)
    pattern_bonus = 0.0
    detected_patterns = []
    
    try:
        from find_high_compression import detect_all_patterns
        patterns = detect_all_patterns(df)
        if patterns:
            best_confidence = patterns[0][1]
            pattern_bonus = best_confidence * 1.0
            detected_patterns = [(p[0], round(p[1], 2)) for p in patterns]
    except ImportError:
        pass
    
    results["pattern_detection"] = {
        "patterns": detected_patterns,
        "bonus_score": round(pattern_bonus, 2),
        "weight": "10%"
    }
    
    results["volatility_squeeze_bonus"] = {
        "score": round(volatility_bonus, 2),
        "weight": "10%",
        "details": {
            "ttm_squeeze": results["ttm_squeeze"]["squeeze_on"],
            "atr_compressed": results["atr_compression"]["compressed"],
            "bbw_squeeze": results["bbw_squeeze"]["is_squeeze"]
        }
    }
    
    # ============================================================
    # MOMENTUM INDICATORS (Paper-based "Sniper" Scoring)
    # ============================================================
    momentum_bonus = 0.0
    try:
        from momentum_indicators import (
            calculate_macd, calculate_stochastic, detect_rsi_divergence,
            calculate_obv_trend, detect_volume_surge, calculate_supertrend,
            calculate_vwap
        )
        
        # Calculate all momentum indicators
        macd = calculate_macd(df)
        stochastic = calculate_stochastic(df)
        rsi_div = detect_rsi_divergence(df)
        obv = calculate_obv_trend(df)
        volume = detect_volume_surge(df)
        supertrend = calculate_supertrend(df)
        vwap = calculate_vwap(df)
        
        # Sum up momentum contributions (capped at 3.0 points to not overwhelm compression score)
        raw_momentum = (
            macd['score_contribution'] +
            stochastic['score_contribution'] +
            rsi_div['score_contribution'] +
            obv['score_contribution'] +
            volume['score_contribution'] +
            supertrend['score_contribution'] +
            vwap['score_contribution']
        )
        
        # Normalize: max raw is ~9, we want max 3 points contribution
        momentum_bonus = min(3.0, raw_momentum / 3.0)
        
        # Apply TTM squeeze penalty if active
        if results["ttm_squeeze"]["squeeze_on"]:
            momentum_bonus -= 0.5
            momentum_bonus = max(0, momentum_bonus)
        
        results["momentum_indicators"] = {
            "macd": {
                "trend": macd["trend"],
                "crossover": macd["crossover"],
                "score": macd["score_contribution"]
            },
            "stochastic": {
                "k": stochastic["k"],
                "d": stochastic["d"],
                "signal": stochastic["signal"],
                "zone": stochastic["zone"],
                "score": stochastic["score_contribution"]
            },
            "rsi": {
                "value": rsi_div["rsi_value"],
                "divergence": rsi_div["divergence_type"],
                "strength": rsi_div["divergence_strength"],
                "score": rsi_div["score_contribution"]
            },
            "obv": {
                "trend": obv["trend"],
                "divergence": obv["divergence"],
                "score": obv["score_contribution"]
            },
            "volume": {
                "rvol": volume["rvol"],
                "is_surge": volume["is_surge"],
                "surge_type": volume["surge_type"],
                "score": volume["score_contribution"]
            },
            "supertrend": {
                "trend": supertrend["trend"],
                "buy_signal": supertrend["buy_signal"],
                "score": supertrend["score_contribution"]
            },
            "vwap": {
                "value": vwap["vwap"],
                "position": vwap["position"],
                "distance_pct": vwap["distance_pct"],
                "score": vwap["score_contribution"]
            },
            "total_momentum_bonus": round(momentum_bonus, 2)
        }
        
    except ImportError:
        # Momentum indicators module not available - graceful fallback
        results["momentum_indicators"] = {"available": False}
    except Exception as e:
        results["momentum_indicators"] = {"error": str(e)}
    
    # Total score (0-10 scale) - now includes momentum bonus
    total_score = core_score + volatility_bonus + pattern_bonus + momentum_bonus
    score = round(min(10.0, total_score), 1)  # Cap at 10
    
    # Store score breakdown
    results["score_breakdown"] = {
        "core_score": round(core_score, 2),
        "volatility_bonus": round(volatility_bonus, 2),
        "pattern_bonus": round(pattern_bonus, 2),
        "momentum_bonus": round(momentum_bonus, 2),
        "total": score,
        "max_possible": 10
    }

    # Add entry/exit timing analysis
    entry_exit_analysis = analyze_entry_exit_timing(df, highest_high, lowest_low, range_percent)
    results["entry_exit_timing"] = entry_exit_analysis

    return {"score": score, "results": results}

def analyze_entry_exit_timing(df: pd.DataFrame, boundary_high: float, boundary_low: float, range_percent: float) -> dict:
    """
    Analyze optimal entry and exit timing for compression patterns.
    """
    if len(df) < 5:
        return {"entry_time": None, "exit_time": None, "optimal_entry_price": None, "optimal_exit_price": None}
    
    # Current price (last close)
    current_price = df['close'].iloc[-1]
    
    # Entry timing: Best entry point during compression
    # Look for lowest volume day during compression (accumulation phase)
    compression_phase = df.copy()
    low_volume_days = compression_phase[compression_phase['volume'] == compression_phase['volume'].quantile(0.25)]
    
    if len(low_volume_days) > 0:
        optimal_entry_day = low_volume_days['close'].idxmin()
        optimal_entry_price = low_volume_days['close'].min()
    else:
        optimal_entry_day = df.index[-5]  # 5 days ago as fallback
        optimal_entry_price = df['close'].iloc[-5]
    
    # Exit timing: Calculate targets based on range expansion
    range_width = boundary_high - boundary_low
    upside_target = boundary_high + (range_width * 0.5)  # 50% of range above high
    downside_target = boundary_low - (range_width * 0.5)  # 50% of range below low
    
    # Risk management: Stop loss levels
    upside_stop = boundary_low - (range_width * 0.1)  # 10% below low
    downside_stop = boundary_high + (range_width * 0.1)  # 10% above high
    
    # Determine current position relative to boundaries
    if current_price > boundary_high:
        current_position = "UPWARD_BREAKOUT"
        recommended_action = "HOLD or PARTIAL_EXIT"
    elif current_price < boundary_low:
        current_position = "DOWNWARD_BREAKOUT"
        recommended_action = "HOLD or PARTIAL_EXIT"
    else:
        current_position = "IN_COMPRESSION"
        recommended_action = "WAIT_FOR_BREAKOUT"
    
    return {
        "optimal_entry_time": optimal_entry_day.strftime('%Y-%m-%d') if optimal_entry_day else None,
        "optimal_entry_price": round(optimal_entry_price, 2) if optimal_entry_price else None,
        "upside_target": round(upside_target, 2),
        "downside_target": round(downside_target, 2),
        "upside_stop": round(upside_stop, 2),
        "downside_stop": round(downside_stop, 2),
        "current_position": current_position,
        "recommended_action": recommended_action,
        "range_width": round(range_width, 2),
        "range_percentage": round(range_percent, 2),
        "boundary_high": round(boundary_high, 2),
        "boundary_low": round(boundary_low, 2)
    }