"""
Advanced chaos statistics computed at high speed.

This module provides several physics-inspired indicators that help detect
market regime shifts. All heavy numerical loops are accelerated with Numba
to keep latency close to C++ implementations while remaining within a Python
tooling stack for research productivity.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numba import jit


ArrayLike = Iterable[float]


@jit(nopython=True)
def _hurst_rs_core(values: np.ndarray) -> float:
    n = values.shape[0]
    if n < 2:
        return np.nan

    mean = 0.0
    for i in range(n):
        mean += values[i]
    mean /= n

    cumulative = 0.0
    max_cumulative = 0.0
    min_cumulative = 0.0
    sum_sq = 0.0

    for i in range(n):
        diff = values[i] - mean
        cumulative += diff
        if cumulative > max_cumulative:
            max_cumulative = cumulative
        if cumulative < min_cumulative:
            min_cumulative = cumulative
        sum_sq += diff * diff

    std = math.sqrt(sum_sq / n)
    if std <= 1e-12:
        return 0.5

    rs = (max_cumulative - min_cumulative) / std
    if rs <= 0.0:
        return 0.5

    return math.log(rs) / math.log(n)


@jit(nopython=True)
def _variogram_regression(values: np.ndarray, max_scale: int) -> float:
    n = values.shape[0]
    usable = min(max_scale, n // 2)
    if usable < 2:
        return np.nan

    log_lags = np.empty(usable, dtype=np.float64)
    log_grams = np.empty(usable, dtype=np.float64)

    for idx in range(usable):
        lag = idx + 1
        count = n - lag
        accum = 0.0
        for j in range(count):
            diff = values[j + lag] - values[j]
            if diff < 0:
                diff = -diff
            accum += diff
        mean_abs = accum / count
        if mean_abs <= 0:
            mean_abs = 1e-12
        log_lags[idx] = math.log(lag)
        log_grams[idx] = math.log(mean_abs)

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for i in range(usable):
        x = log_lags[i]
        y = log_grams[i]
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denominator = usable * sum_x2 - sum_x * sum_x
    if abs(denominator) <= 1e-12:
        return np.nan

    slope = (usable * sum_xy - sum_x * sum_y) / denominator
    return 2.0 - slope


def _as_float_array(series: ArrayLike) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    arr = arr[mask]
    return arr


def calculate_hurst_rs(series: ArrayLike) -> float:
    """
    Calculate the Hurst exponent using Rescaled Range analysis.

    Parameters
    ----------
    series : Iterable[float]
        Time series values (e.g., price levels).

    Returns
    -------
    float
        Hurst exponent. NaN when series is too short or degenerate.
    """
    values = _as_float_array(series)
    if values.size < 2:
        return float("nan")
    return float(_hurst_rs_core(values))


def calculate_shannon_entropy(series: ArrayLike, bins: int = 50) -> float:
    """
    Calculate Shannon entropy after discretizing returns.

    Parameters
    ----------
    series : Iterable[float]
        Input observations (e.g., rolling returns or price differences).
    bins : int, optional
        Number of histogram buckets used for discretization.

    Returns
    -------
    float
        Estimated Shannon entropy in nats.
    """
    values = _as_float_array(series)
    if values.size < 2 or bins < 2:
        return float("nan")

    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        return float("nan")
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy)


def calculate_fdi(series: ArrayLike, max_scale: int = 8) -> float:
    """
    Estimate the Fractal Dimension Index using a variogram proxy.

    Parameters
    ----------
    series : Iterable[float]
        Price or indicator series.
    max_scale : int, optional
        Maximum lag to include in the log-log regression.

    Returns
    -------
    float
        Fractal Dimension Index where values near 1.0 indicate trending
        behavior and values near 2.0 suggest noisy, mean-reverting regimes.
    """
    values = _as_float_array(series)
    if values.size < 8:
        return float("nan")
    return float(_variogram_regression(values, max_scale))


def _select_price_column(df: pd.DataFrame) -> pd.Series:
    candidate_columns: Sequence[str] = (
        "close",
        "price",
        "settle",
        "value",
    )
    for col in candidate_columns:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    if numeric_cols.empty:
        raise ValueError(
            "No numeric columns available to compute physics indicators."
        )
    return pd.to_numeric(df[numeric_cols[0]], errors="coerce")


def apply_rolling_physics(
    df: pd.DataFrame, windows: Sequence[int] = (100, 200)
) -> pd.DataFrame:
    """
    Apply rolling physics indicators across multiple lookback windows.
    
    For multi-asset data (with 'asset_id' column), calculates indicators
    per-asset to prevent feature bleeding between different cryptocurrencies.

    Parameters
    ----------
    df : pd.DataFrame
        Input market data. A numeric column named 'close', 'price', or similar
        is preferred; otherwise the first numeric column is used.
        If 'asset_id' column exists, calculations are done per-asset.
    windows : Sequence[int], optional
        Window sizes to evaluate. Defaults to (100, 200).

    Returns
    -------
    pd.DataFrame
        Copy of the original frame with additional chaos metrics appended.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(windows, Sequence) or len(windows) == 0:
        raise ValueError("At least one window length must be provided.")

    # Check if multi-asset mode
    if 'asset_id' in df.columns:
        print("    [PHYSICS] Multi-asset mode detected, calculating per asset...")
        groups = []
        for asset_id in sorted(df['asset_id'].unique()):
            asset_df = df[df['asset_id'] == asset_id].copy()
            print(f"      Processing asset_id={asset_id} ({len(asset_df)} rows)...")
            
            # Calculate physics for this asset
            asset_df = _calculate_physics_single_asset(asset_df, windows)
            groups.append(asset_df)
        
        result = pd.concat(groups).sort_index()
        print(f"    [PHYSICS] Completed for {len(groups)} assets")
        return result
    else:
        # Single-asset mode
        return _calculate_physics_single_asset(df, windows)


def _calculate_physics_single_asset(
    df: pd.DataFrame, windows: Sequence[int]
) -> pd.DataFrame:
    """
    Calculate physics indicators for a single asset.
    
    Internal helper function used by apply_rolling_physics.
    """
    price_series = _select_price_column(df)
    result = df.copy()

    def _hurst_func(window_values: np.ndarray) -> float:
        return calculate_hurst_rs(window_values)

    def _entropy_func(window_values: np.ndarray) -> float:
        returns = np.diff(window_values)
        return calculate_shannon_entropy(returns)

    def _fdi_func(window_values: np.ndarray) -> float:
        return calculate_fdi(window_values)

    for window in windows:
        if window < 10:
            raise ValueError("Window length must be at least 10 observations.")

        hurst_col = f"hurst_{window}"
        entropy_col = f"entropy_{window}"
        fdi_col = f"fdi_{window}"

        result[hurst_col] = price_series.rolling(
            window=window, min_periods=window
        ).apply(_hurst_func, raw=True)

        result[entropy_col] = price_series.rolling(
            window=window, min_periods=window
        ).apply(_entropy_func, raw=True)

        result[fdi_col] = price_series.rolling(
            window=window, min_periods=window
        ).apply(_fdi_func, raw=True)

    return result


__all__ = [
    "calculate_hurst_rs",
    "calculate_shannon_entropy",
    "calculate_fdi",
    "apply_rolling_physics",
    "apply_stability_physics",
]


def apply_stability_physics(
    df: pd.DataFrame,
    window: int = 168,
    z_window: int = 504,  # ~3x the calculation window for robust Z-scores
    threshold_z: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate physics-based stability metrics (Theta, ACF) and detect warnings.
    
    Migrated from StabilityMonitor (Stage 3) for production use.
    Uses vectorized pandas operations for performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a price column (close/price/value).
    window : int, default=168
        Rolling window size for stability metrics (e.g., 168 hours = 1 week).
    z_window : int, default=504
        Rolling window for Z-score calculation (history to compare against).
    threshold_z : float, default=2.0
        Z-score threshold for triggering stability warnings.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with appended columns:
        - stability_acf: Lag-1 Autocorrelation
        - stability_theta: Mean reversion speed (Ornstein-Uhlenbeck)
        - stability_warning: 1 if unstable (High ACF or Low Theta), 0 otherwise
    """
    # Select price series
    prices = _select_price_column(df)
    
    # Pre-calculate lagged series for vectorization
    # x_t (Predictor) -> prices.shift(1)
    # x_t+1 (Target)  -> prices
    lagged = prices.shift(1)
    
    # 1. Calculate Rolling Lag-1 Autocorrelation (ACF)
    # corr(x_t, x_{t-1}) over the window
    stability_acf = prices.rolling(window=window).corr(lagged)
    
    # 2. Calculate Rolling Dynamic Theta (OU Process)
    # Model: x_{t+1} = alpha + beta * x_t + epsilon
    # beta = Cov(x_{t+1}, x_t) / Var(x_t)
    # theta = -ln(beta) / dt (assuming dt=1)
    
    rolling_cov = prices.rolling(window=window).cov(lagged)
    rolling_var = lagged.rolling(window=window).var()
    
    # Avoid division by zero
    beta = rolling_cov / rolling_var.replace(0, np.nan)
    
    # Calculate Theta: -log(beta)
    # If beta <= 0 or beta >= 1, theta is effectively 0 (critical/unstable)
    # We clip beta to (0, 1) for the log calculation, then handle edge cases
    beta_clipped = beta.clip(lower=1e-6, upper=1.0 - 1e-6)
    stability_theta = -np.log(beta_clipped)
    
    # Set Theta to 0 where beta was out of bounds (non-mean-reverting)
    mask_unstable = (beta <= 0) | (beta >= 1.0)
    stability_theta[mask_unstable] = 0.0
    
    # 3. Calculate Z-scores for Warning Detection
    # Z = (Value - Mean) / Std
    
    # ACF Z-score (Rising ACF is bad)
    acf_mean = stability_acf.rolling(window=z_window, min_periods=window).mean()
    acf_std = stability_acf.rolling(window=z_window, min_periods=window).std()
    z_acf = (stability_acf - acf_mean) / acf_std.replace(0, 1e-9)
    
    # Theta Z-score (Falling Theta is bad)
    theta_mean = stability_theta.rolling(window=z_window, min_periods=window).mean()
    theta_std = stability_theta.rolling(window=z_window, min_periods=window).std()
    z_theta = (stability_theta - theta_mean) / theta_std.replace(0, 1e-9)
    
    # 4. Generate Warnings
    # Warning if ACF spikes (Z > threshold) OR Theta drops (Z < -threshold)
    warning_acf = z_acf > threshold_z
    warning_theta = z_theta < -threshold_z
    
    stability_warning = (warning_acf | warning_theta).astype(int)
    
    # Append to DataFrame
    result = df.copy()
    result["stability_acf"] = stability_acf.astype(np.float32).fillna(0)
    result["stability_theta"] = stability_theta.astype(np.float32).fillna(0)
    result["stability_warning"] = stability_warning.astype(np.int32).fillna(0)
    
    return result
