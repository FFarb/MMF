"""
Memory-Preserving Fractional Differentiation.

This module implements Fractional Differentiation (FracDiff) to achieve stationarity
while retaining long-term memory in time series data. Unlike standard differencing
(d=1.0) which destroys trend information, fractional differentiation (d ≈ 0.4)
preserves memory while making the series stationary.

References
----------
- Marcos López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 5
- Hosking, J.R.M. (1981). "Fractional Differencing"

Mathematical Foundation
-----------------------
The fractional difference operator is defined as:

    X_t^d = Σ(k=0 to ∞) w_k * X_{t-k}

where the weights follow the iterative formula:

    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

For d ∈ [0, 1]:
- d = 0.0: No transformation (original series)
- d = 0.4-0.6: Optimal balance (stationary + memory)
- d = 1.0: Standard differencing (stationary, no memory)
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from numba import jit
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


@jit(nopython=True, cache=True)
def _compute_weights_numba(d: float, size: int) -> np.ndarray:
    """
    Compute fractional differentiation weights using Numba acceleration.
    
    This implements the iterative formula:
        w_k = -w_{k-1} * (d - k + 1) / k
    
    Parameters
    ----------
    d : float
        Differentiation order (0 < d < 1 for fractional)
    size : int
        Number of weights to compute
    
    Returns
    -------
    np.ndarray
        Array of weights with shape (size,)
    """
    weights = np.zeros(size, dtype=np.float64)
    weights[0] = 1.0
    
    for k in range(1, size):
        weights[k] = -weights[k - 1] * (d - k + 1) / k
    
    return weights


@jit(nopython=True, cache=True)
def _apply_weights_numba(
    series: np.ndarray,
    weights: np.ndarray,
    threshold: float = 1e-5
) -> np.ndarray:
    """
    Apply fractional differentiation weights to a series using fixed window.
    
    Uses strict causality - only past values are used (no future leakage).
    Weights below threshold are truncated for computational efficiency.
    
    Parameters
    ----------
    series : np.ndarray
        Input time series (1D array)
    weights : np.ndarray
        Fractional differentiation weights
    threshold : float
        Minimum absolute weight value to consider (for truncation)
    
    Returns
    -------
    np.ndarray
        Fractionally differentiated series (same length as input, with NaNs at start)
    """
    n = len(series)
    w_len = len(weights)
    
    # Truncate weights below threshold
    cutoff = w_len
    for i in range(w_len):
        if abs(weights[i]) < threshold:
            cutoff = i
            break
    
    weights_trunc = weights[:cutoff]
    result = np.full(n, np.nan, dtype=np.float64)
    
    # Apply weights using fixed window (strict causality)
    for t in range(cutoff - 1, n):
        val = 0.0
        for k in range(len(weights_trunc)):
            if t - k >= 0:
                val += weights_trunc[k] * series[t - k]
        result[t] = val
    
    return result


class FractionalDifferentiator:
    """
    Production-grade Fractional Differentiation for time series.
    
    This class provides memory-preserving stationarity transformation using
    fractional differentiation. It supports:
    - Configurable memory window for computational efficiency
    - Automatic optimal d-value discovery via ADF test
    - Strict causality (no future leakage)
    - Numba-accelerated computation
    
    Attributes
    ----------
    window_size : int
        Maximum number of past observations to use (memory window)
    weights_cache : dict
        Cache of computed weights for different d values
    optimal_d_ : float or None
        Optimal d value found by find_min_d (if called)
    
    Examples
    --------
    >>> # Basic usage
    >>> frac_diff = FractionalDifferentiator(window_size=2048)
    >>> series_diff = frac_diff.transform(price_series, d=0.4)
    
    >>> # Find optimal d for stationarity
    >>> optimal_d = frac_diff.find_min_d(price_series, precision=0.01)
    >>> series_stationary = frac_diff.transform(price_series, d=optimal_d)
    """
    
    def __init__(self, window_size: int = 2048):
        """
        Initialize the Fractional Differentiator.
        
        Parameters
        ----------
        window_size : int, default=2048
            Maximum lookback window for weight calculation.
            Larger values preserve more memory but increase computation time.
            Typical values: 512-4096 for financial data.
        """
        if window_size < 10:
            raise ValueError("window_size must be at least 10")
        
        self.window_size = window_size
        self.weights_cache = {}
        self.optimal_d_ = None
    
    def _get_weights(self, d: float, size: Optional[int] = None) -> np.ndarray:
        """
        Get fractional differentiation weights (with caching).
        
        Implements the iterative formula:
            w_k = -w_{k-1} * (d - k + 1) / k
        
        Parameters
        ----------
        d : float
            Differentiation order (0 <= d <= 1)
        size : int, optional
            Number of weights to compute. If None, uses self.window_size
        
        Returns
        -------
        np.ndarray
            Array of weights
        """
        if size is None:
            size = self.window_size
        
        # Check cache
        cache_key = (d, size)
        if cache_key in self.weights_cache:
            return self.weights_cache[cache_key]
        
        # Compute weights using Numba
        weights = _compute_weights_numba(d, size)
        
        # Cache for reuse
        self.weights_cache[cache_key] = weights
        
        return weights
    
    def transform(
        self,
        series: Union[pd.Series, np.ndarray],
        d: float,
        threshold: float = 1e-5,
        drop_na: bool = False
    ) -> Union[pd.Series, np.ndarray]:
        """
        Apply fractional differentiation to a time series.
        
        Uses a fixed window method to ensure strict causality (no future leakage).
        The first (window_size - 1) values will be NaN.
        
        Parameters
        ----------
        series : pd.Series or np.ndarray
            Input time series to transform
        d : float
            Differentiation order:
            - d = 0.0: No transformation
            - d = 0.4-0.6: Optimal balance (stationary + memory)
            - d = 1.0: Standard differencing
        threshold : float, default=1e-5
            Minimum absolute weight value to consider (for truncation)
        drop_na : bool, default=False
            If True, drop NaN values at the start of the result
        
        Returns
        -------
        pd.Series or np.ndarray
            Fractionally differentiated series (same type as input)
        """
        if d < 0 or d > 1:
            raise ValueError(f"d must be in [0, 1], got {d}")
        
        # Handle pandas Series
        is_series = isinstance(series, pd.Series)
        if is_series:
            index = series.index
            values = series.values
        else:
            values = np.asarray(series)
        
        # Handle edge cases
        if len(values) < 10:
            raise ValueError("Series must have at least 10 observations")
        
        if d == 0.0:
            # No transformation
            return series.copy() if is_series else values.copy()
        
        # Get weights
        weights = self._get_weights(d, size=min(self.window_size, len(values)))
        
        # Apply transformation using Numba
        result = _apply_weights_numba(values, weights, threshold)
        
        # Convert back to Series if needed
        if is_series:
            result = pd.Series(result, index=index, name=f"{series.name}_fracdiff_{d:.2f}")
            if drop_na:
                result = result.dropna()
        elif drop_na:
            result = result[~np.isnan(result)]
        
        return result
    
    def find_min_d(
        self,
        series: Union[pd.Series, np.ndarray],
        precision: float = 0.01,
        max_d: float = 0.65,  # SOFT CAP: Prevent memory burnout
        adf_pvalue_threshold: float = 0.05,
        verbose: bool = True
    ) -> float:
        """
        Find the minimum d value that makes the series stationary.
        
        Iterates d from 0.0 to max_d and applies the Augmented Dickey-Fuller (ADF)
        test to determine stationarity. Returns the minimum d where p-value < threshold,
        OR max_d if no stationary d is found (accepting "stationary enough").
        
        **ADAPTIVE MEMORY PATCH**: Default max_d=0.65 prevents aggressive differentiation
        that destroys market memory on volatile assets. Better to preserve some signal
        with slight non-stationarity than achieve perfect stationarity with no memory.
        
        This is useful for finding the optimal balance between stationarity and
        memory preservation.
        
        Parameters
        ----------
        series : pd.Series or np.ndarray
            Input time series
        precision : float, default=0.01
            Step size for d iteration (smaller = more precise but slower)
        max_d : float, default=0.65
            **SOFT CAP**: Maximum d value to test. Assets that require d > 0.65
            for strict stationarity will be capped here, preserving memory.
            Original default was 1.0 (too aggressive for altcoins).
        adf_pvalue_threshold : float, default=0.05
            ADF test p-value threshold for stationarity
        verbose : bool, default=True
            If True, print progress information
        
        Returns
        -------
        float
            Optimal d value (minimum d that achieves stationarity)
            Returns max_d if no stationary d is found (Soft Cap applied)
        
        Notes
        -----
        The ADF test null hypothesis is that the series has a unit root (non-stationary).
        We reject the null (conclude stationarity) when p-value < threshold.
        
        **Why Soft Cap?**
        Diagnostics showed that volatile altcoins (e.g., ADAUSDT) required d ≈ 1.0
        for strict stationarity, which destroyed memory (correlation < 0.25).
        By capping at d=0.65, we preserve ~40-60% correlation, enabling Trend Expert
        to function while accepting slight non-stationarity.
        """
        if verbose:
            print(f"\n[FracDiff] Finding optimal d for stationarity...")
            print(f"  Precision: {precision}, Max d (SOFT CAP): {max_d}, ADF threshold: {adf_pvalue_threshold}")
        
        # Test original series first
        try:
            adf_result = adfuller(series, maxlag=1, regression='c', autolag=None)
            original_pvalue = adf_result[1]
            
            if verbose:
                print(f"  Original series ADF p-value: {original_pvalue:.4f}")
            
            if original_pvalue < adf_pvalue_threshold:
                if verbose:
                    print(f"  [OK] Series is already stationary (d=0.0)")
                self.optimal_d_ = 0.0
                return 0.0
        except Exception as e:
            if verbose:
                print(f"  Warning: ADF test on original series failed: {e}")
        
        # Iterate through d values
        d_values = np.arange(precision, max_d + precision, precision)
        
        for d in d_values:
            try:
                # Transform series
                series_diff = self.transform(series, d=d, drop_na=True)
                
                # Skip if too few observations after dropping NaNs
                if len(series_diff) < 50:
                    continue
                
                # Run ADF test
                adf_result = adfuller(series_diff, maxlag=1, regression='c', autolag=None)
                pvalue = adf_result[1]
                
                if verbose and (d * 100) % 10 == 0:  # Print every 0.1
                    print(f"  d={d:.2f}: ADF p-value={pvalue:.4f}")
                
                # Check if stationary OR hit soft cap
                if pvalue < adf_pvalue_threshold:
                    if verbose:
                        print(f"  [OK] Found optimal d={d:.3f} (ADF p-value={pvalue:.4f})")
                    self.optimal_d_ = d
                    return d
                elif d >= max_d - precision:
                    # Hit soft cap - accept this d even if not strictly stationary
                    if verbose:
                        print(f"  [SOFT CAP] Reached max_d={max_d:.3f} (ADF p-value={pvalue:.4f})")
                        print(f"  [SOFT CAP] Accepting d={max_d:.3f} to preserve memory (not strictly stationary)")
                    self.optimal_d_ = max_d
                    return max_d
            
            except Exception as e:
                if verbose:
                    print(f"  Warning: ADF test failed for d={d:.2f}: {e}")
                continue
        
        # Fallback: return max_d (Soft Cap)
        if verbose:
            print(f"  [SOFT CAP] No stationary d found, returning max_d={max_d:.3f}")
            print(f"  [SOFT CAP] Preserving memory over strict stationarity")
        
        self.optimal_d_ = max_d
        return max_d
    
    def fit_transform(
        self,
        series: Union[pd.Series, np.ndarray],
        precision: float = 0.01,
        drop_na: bool = False
    ) -> Union[pd.Series, np.ndarray]:
        """
        Find optimal d and transform in one step.
        
        Convenience method that combines find_min_d and transform.
        
        Parameters
        ----------
        series : pd.Series or np.ndarray
            Input time series
        precision : float, default=0.01
            Step size for d iteration in find_min_d
        drop_na : bool, default=False
            If True, drop NaN values in the result
        
        Returns
        -------
        pd.Series or np.ndarray
            Fractionally differentiated series using optimal d
        """
        optimal_d = self.find_min_d(series, precision=precision)
        return self.transform(series, d=optimal_d, drop_na=drop_na)
    
    def get_info(self) -> dict:
        """
        Get information about the differentiator state.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - window_size: Configured window size
            - cache_size: Number of cached weight arrays
            - optimal_d: Last optimal d found (or None)
        """
        return {
            "window_size": self.window_size,
            "cache_size": len(self.weights_cache),
            "optimal_d": self.optimal_d_,
        }
    
    def clear_cache(self):
        """Clear the weights cache to free memory."""
        self.weights_cache.clear()


def apply_frac_diff_to_dataframe(
    df: pd.DataFrame,
    price_col: str = "close",
    d: Optional[float] = None,
    find_optimal: bool = True,
    precision: float = 0.01,
    window_size: int = 2048,
    output_col: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply fractional differentiation to a DataFrame column.
    
    Convenience function for applying FracDiff to a specific column in a DataFrame.
    Handles multi-asset data (with 'asset_id' column) by processing each asset separately.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str, default='close'
        Column name to apply fractional differentiation to
    d : float, optional
        Differentiation order. If None and find_optimal=True, will find optimal d
    find_optimal : bool, default=True
        If True and d is None, find optimal d using ADF test
    precision : float, default=0.01
        Precision for optimal d search
    window_size : int, default=2048
        Window size for FractionalDifferentiator
    output_col : str, optional
        Name for output column. If None, uses f"{price_col}_fracdiff"
    verbose : bool, default=True
        Print progress information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added fractional differentiation column
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    if output_col is None:
        output_col = f"{price_col}_fracdiff"
    
    df_out = df.copy()
    frac_diff = FractionalDifferentiator(window_size=window_size)
    
    # Check if multi-asset
    has_asset_id = 'asset_id' in df.columns
    
    if has_asset_id:
        # Process each asset separately
        if verbose:
            print(f"[FracDiff] Processing multi-asset data...")
        
        results = []
        for asset_id in df['asset_id'].unique():
            mask = df['asset_id'] == asset_id
            series = df.loc[mask, price_col]
            
            if verbose:
                print(f"\n  Asset {asset_id}:")
            
            # Find or use d
            if d is None and find_optimal:
                d_use = frac_diff.find_min_d(series, precision=precision, verbose=verbose)
            elif d is None:
                d_use = 0.5  # Default
            else:
                d_use = d
            
            # Transform
            series_diff = frac_diff.transform(series, d=d_use)
            results.append(series_diff)
        
        df_out[output_col] = pd.concat(results)
    
    else:
        # Single asset
        series = df[price_col]
        
        # Find or use d
        if d is None and find_optimal:
            d_use = frac_diff.find_min_d(series, precision=precision, verbose=verbose)
        elif d is None:
            d_use = 0.5  # Default
        else:
            d_use = d
        
        # Transform
        df_out[output_col] = frac_diff.transform(series, d=d_use)
    
    return df_out
