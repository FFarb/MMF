"""
Technical Analysis Core Functions.

Pure numpy/pandas implementation of common TA indicators.
Zero external dependencies. Stable on all timeframes.

All functions return pandas Series or DataFrame with the same index as input.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).
    
    Args:
        close: Close price series
        length: RSI period (default: 14)
        
    Returns:
        RSI values (0-100)
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    
    rs = gain / (loss + 1e-10)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def ema(close: pd.Series, length: int) -> pd.Series:
    """
    Exponential Moving Average (EMA).
    
    Args:
        close: Close price series
        length: EMA period
        
    Returns:
        EMA values
    """
    return close.ewm(span=length, adjust=False).mean()


def sma(close: pd.Series, length: int) -> pd.Series:
    """
    Simple Moving Average (SMA).
    
    Args:
        close: Close price series
        length: SMA period
        
    Returns:
        SMA values
    """
    return close.rolling(window=length).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).
    
    Args:
        close: Close price series
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
        
    Returns:
        DataFrame with columns: MACD, MACDh (histogram), MACDs (signal)
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    result = pd.DataFrame(index=close.index)
    result[f'MACD_{fast}_{slow}_{signal}'] = macd_line
    result[f'MACDh_{fast}_{slow}_{signal}'] = histogram
    result[f'MACDs_{fast}_{slow}_{signal}'] = signal_line
    
    return result


def bbands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.
    
    Args:
        close: Close price series
        length: Period for moving average (default: 20)
        std: Standard deviation multiplier (default: 2.0)
        
    Returns:
        DataFrame with columns: BBL (lower), BBM (middle), BBU (upper), BBB (bandwidth), BBP (percent B)
    """
    middle = sma(close, length)
    std_dev = close.rolling(window=length).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    bandwidth = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower + 1e-10)
    
    # Format std as integer if it's a whole number
    std_str = str(int(std)) if std == int(std) else str(std)
    
    result = pd.DataFrame(index=close.index)
    result[f'BBL_{length}_{std_str}'] = lower
    result[f'BBM_{length}_{std_str}'] = middle
    result[f'BBU_{length}_{std_str}'] = upper
    result[f'BBB_{length}_{std_str}'] = bandwidth
    result[f'BBP_{length}_{std_str}'] = percent_b
    
    return result


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14
) -> pd.Series:
    """
    Average True Range (ATR).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: ATR period (default: 14)
        
    Returns:
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_values = true_range.rolling(window=length).mean()
    
    return atr_values


def natr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14
) -> pd.Series:
    """
    Normalized Average True Range (NATR).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: NATR period (default: 14)
        
    Returns:
        NATR values (percentage)
    """
    atr_values = atr(high, low, close, length)
    natr_values = (atr_values / close) * 100
    
    return natr_values


def roc(close: pd.Series, length: int = 10) -> pd.Series:
    """
    Rate of Change (ROC).
    
    Args:
        close: Close price series
        length: ROC period (default: 10)
        
    Returns:
        ROC values (percentage)
    """
    roc_values = ((close - close.shift(length)) / close.shift(length)) * 100
    
    return roc_values


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    constant: float = 0.015
) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: CCI period (default: 20)
        constant: Scaling constant (default: 0.015)
        
    Returns:
        CCI values
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=length).mean()
    mean_deviation = typical_price.rolling(window=length).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    
    cci_values = (typical_price - sma_tp) / (constant * mean_deviation + 1e-10)
    
    return cci_values


def willr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14
) -> pd.Series:
    """
    Williams %R.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: Williams %R period (default: 14)
        
    Returns:
        Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    
    willr_values = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    return willr_values


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14
) -> pd.DataFrame:
    """
    Average Directional Index (ADX).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: ADX period (default: 14)
        
    Returns:
        DataFrame with columns: ADX, DMP (plus), DMN (minus)
    """
    # Calculate True Range
    tr = atr(high, low, close, 1)
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Smoothed indicators
    atr_smooth = tr.rolling(window=length).mean()
    plus_di = 100 * (plus_dm.rolling(window=length).mean() / (atr_smooth + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=length).mean() / (atr_smooth + 1e-10))
    
    # ADX calculation
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = dx.rolling(window=length).mean()
    
    result = pd.DataFrame(index=close.index)
    result[f'ADX_{length}'] = adx_values
    result[f'DMP_{length}'] = plus_di
    result[f'DMN_{length}'] = minus_di
    
    return result


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV).
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        OBV values
    """
    direction = np.sign(close.diff())
    obv_values = (direction * volume).fillna(0).cumsum()
    
    return obv_values


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 20
) -> pd.Series:
    """
    Chaikin Money Flow (CMF).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        length: CMF period (default: 20)
        
    Returns:
        CMF values
    """
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * volume
    
    cmf_values = mfv.rolling(window=length).sum() / (volume.rolling(window=length).sum() + 1e-10)
    
    return cmf_values


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14
) -> pd.Series:
    """
    Money Flow Index (MFI).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        length: MFI period (default: 14)
        
    Returns:
        MFI values (0-100)
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    
    positive_mf = positive_flow.rolling(window=length).sum()
    negative_mf = negative_flow.rolling(window=length).sum()
    
    mfi_ratio = positive_mf / (negative_mf + 1e-10)
    mfi_values = 100 - (100 / (1 + mfi_ratio))
    
    return mfi_values


def slope(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Linear regression slope.
    
    Args:
        close: Close price series
        length: Period for slope calculation (default: 14)
        
    Returns:
        Slope values
    """
    def calc_slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope_val = np.polyfit(x, y, 1)[0]
        return slope_val
    
    slope_values = close.rolling(window=length).apply(calc_slope, raw=True)
    
    return slope_values


def hurst(close: pd.Series, length: int = 100) -> pd.Series:
    """
    Hurst Exponent (R/S method).
    
    Args:
        close: Close price series
        length: Period for Hurst calculation (default: 100)
        
    Returns:
        Hurst exponent values (0-1)
    """
    def calc_hurst(prices):
        if len(prices) < 2:
            return np.nan
        
        # Log returns
        log_returns = np.log(prices / np.roll(prices, 1))[1:]
        
        if len(log_returns) < 2:
            return np.nan
        
        # Mean-adjusted series
        mean_adj = log_returns - np.mean(log_returns)
        
        # Cumulative sum
        cumsum = np.cumsum(mean_adj)
        
        # Range
        R = np.max(cumsum) - np.min(cumsum)
        
        # Standard deviation
        S = np.std(log_returns)
        
        if S == 0 or R == 0:
            return 0.5
        
        # R/S ratio
        rs = R / S
        
        # Hurst exponent approximation
        hurst_val = np.log(rs) / np.log(len(log_returns))
        
        return np.clip(hurst_val, 0, 1)
    
    hurst_values = close.rolling(window=length).apply(calc_hurst, raw=True)
    
    return hurst_values


def ao(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
    """
    Awesome Oscillator (AO).
    
    Args:
        high: High price series
        low: Low price series
        fast: Fast period (default: 5)
        slow: Slow period (default: 34)
        
    Returns:
        AO values
    """
    median_price = (high + low) / 2
    ao_values = sma(median_price, fast) - sma(median_price, slow)
    
    return ao_values


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3
) -> pd.DataFrame:
    """
    Stochastic Oscillator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k: %K period (default: 14)
        d: %D period (default: 3)
        smooth_k: %K smoothing (default: 3)
        
    Returns:
        DataFrame with columns: STOCHk, STOCHd
    """
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_k_smooth = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k_smooth.rolling(window=d).mean()
    
    result = pd.DataFrame(index=close.index)
    result[f'STOCHk_{k}_{d}_{smooth_k}'] = stoch_k_smooth
    result[f'STOCHd_{k}_{d}_{smooth_k}'] = stoch_d
    
    return result


def donchian(
    high: pd.Series,
    low: pd.Series,
    lower_length: int = 20,
    upper_length: int = 20
) -> pd.DataFrame:
    """
    Donchian Channels.
    
    Args:
        high: High price series
        low: Low price series
        lower_length: Lower channel period (default: 20)
        upper_length: Upper channel period (default: 20)
        
    Returns:
        DataFrame with columns: DCL (lower), DCM (middle), DCU (upper)
    """
    upper = high.rolling(window=upper_length).max()
    lower = low.rolling(window=lower_length).min()
    middle = (upper + lower) / 2
    
    result = pd.DataFrame(index=high.index)
    result[f'DCL_{lower_length}_{upper_length}'] = lower
    result[f'DCM_{lower_length}_{upper_length}'] = middle
    result[f'DCU_{lower_length}_{upper_length}'] = upper
    
    return result
