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

    Parameters
    ----------
    df : pd.DataFrame
        Input market data. A numeric column named 'close', 'price', or similar
        is preferred; otherwise the first numeric column is used.
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
]
