"""
Physics Profiler - Market Regime Diagnostic Tool

This module provides a comprehensive diagnostic framework to profile the
"physics" of different assets, helping identify why strategies perform
differently across markets.

Key Metrics:
- Hurst Exponent: Trend persistence (H > 0.5 = trending, H < 0.5 = mean-reverting)
- Stability Theta (OU): Mean reversion speed (higher = faster reversion)
- Sample Entropy: Randomness/predictability level
- FracDiff Correlation: Memory retention at optimal d
- Volatility of Volatility: Explosive move detection

Use Case:
Compare BTC vs ADA to understand why Trend Expert works on BTC but fails on ADA.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from ..features.advanced_stats import calculate_hurst_rs, calculate_shannon_entropy
from ..preprocessing.frac_diff import FractionalDifferentiator
from .ou_engine import OUEstimator


class PhysicsProfiler:
    """
    Comprehensive market physics profiler.
    
    Computes distributional statistics of core regime-detection metrics
    to characterize the fundamental behavior of an asset.
    
    Attributes
    ----------
    results_ : Dict[str, Dict[str, float]]
        Nested dictionary containing metric distributions:
        {
            'hurst': {'mean': ..., 'median': ..., 'std': ..., 'min': ..., 'max': ...},
            'theta': {...},
            'entropy': {...},
            'fracdiff_corr': {...},
            'vol_of_vol': {...}
        }
    rolling_results_ : pd.DataFrame
        Time series of all computed metrics for visualization
    """
    
    def __init__(
        self,
        hurst_window: int = 200,
        theta_window: int = 168,
        entropy_window: int = 100,
        fracdiff_window: int = 2048,
        vol_window: int = 24,
    ) -> None:
        """
        Initialize the profiler with window sizes for rolling calculations.
        
        Parameters
        ----------
        hurst_window : int, default=200
            Window size for Hurst exponent calculation
        theta_window : int, default=168
            Window size for OU theta calculation (168h = 1 week)
        entropy_window : int, default=100
            Window size for sample entropy
        fracdiff_window : int, default=2048
            Window size for fractional differentiation
        vol_window : int, default=24
            Window size for volatility calculations
        """
        self.hurst_window = hurst_window
        self.theta_window = theta_window
        self.entropy_window = entropy_window
        self.fracdiff_window = fracdiff_window
        self.vol_window = vol_window
        
        self.results_: Dict[str, Dict[str, float]] = {}
        self.rolling_results_: pd.DataFrame = pd.DataFrame()
        self.optimal_d_: float = 0.0
        
    def fit(self, price_series: pd.Series, verbose: bool = True) -> PhysicsProfiler:
        """
        Compute all physics metrics for the given price series.
        
        Parameters
        ----------
        price_series : pd.Series
            Price time series (e.g., close prices)
        verbose : bool, default=True
            Print progress messages
            
        Returns
        -------
        self : PhysicsProfiler
            Fitted profiler instance
        """
        if verbose:
            print(f"\n[PhysicsProfiler] Analyzing {len(price_series)} data points...")
        
        # Initialize results DataFrame
        results_df = pd.DataFrame(index=price_series.index)
        
        # 1. Rolling Hurst Exponent
        if verbose:
            print(f"  [1/5] Computing Hurst Exponent (window={self.hurst_window})...")
        
        hurst_values = self._compute_rolling_hurst(price_series)
        results_df['hurst'] = hurst_values
        
        # 2. Rolling Theta (OU Mean Reversion Speed)
        if verbose:
            print(f"  [2/5] Computing Stability Theta (window={self.theta_window})...")
        
        theta_values = self._compute_rolling_theta(price_series)
        results_df['theta'] = theta_values
        
        # 3. Rolling Sample Entropy
        if verbose:
            print(f"  [3/5] Computing Sample Entropy (window={self.entropy_window})...")
        
        entropy_values = self._compute_rolling_entropy(price_series)
        results_df['entropy'] = entropy_values
        
        # 4. FracDiff Correlation (Memory Retention)
        if verbose:
            print(f"  [4/5] Computing FracDiff Correlation...")
        
        fracdiff_corr = self._compute_fracdiff_correlation(price_series)
        results_df['fracdiff_corr'] = fracdiff_corr
        
        # 5. Volatility of Volatility
        if verbose:
            print(f"  [5/5] Computing Volatility of Volatility (window={self.vol_window})...")
        
        vol_of_vol = self._compute_vol_of_vol(price_series)
        results_df['vol_of_vol'] = vol_of_vol
        
        # Store rolling results
        self.rolling_results_ = results_df
        
        # Compute distributional statistics
        self.results_ = {}
        for metric in ['hurst', 'theta', 'entropy', 'fracdiff_corr', 'vol_of_vol']:
            clean_values = results_df[metric].dropna()
            
            if len(clean_values) > 0:
                self.results_[metric] = {
                    'mean': float(clean_values.mean()),
                    'median': float(clean_values.median()),
                    'std': float(clean_values.std()),
                    'min': float(clean_values.min()),
                    'max': float(clean_values.max()),
                    'q25': float(clean_values.quantile(0.25)),
                    'q75': float(clean_values.quantile(0.75)),
                    'count': int(len(clean_values)),
                }
            else:
                self.results_[metric] = {
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'q25': np.nan,
                    'q75': np.nan,
                    'count': 0,
                }
        
        if verbose:
            print(f"  [OK] Physics profiling complete!")
        
        return self
    
    def _compute_rolling_hurst(self, price_series: pd.Series) -> pd.Series:
        """Compute rolling Hurst exponent."""
        def hurst_func(window_values: np.ndarray) -> float:
            try:
                return calculate_hurst_rs(window_values)
            except:
                return np.nan
        
        return price_series.rolling(
            window=self.hurst_window,
            min_periods=self.hurst_window
        ).apply(hurst_func, raw=True)
    
    def _compute_rolling_theta(self, price_series: pd.Series) -> pd.Series:
        """Compute rolling OU theta (mean reversion speed)."""
        theta_values = []
        
        for i in range(len(price_series)):
            if i < self.theta_window:
                theta_values.append(np.nan)
            else:
                window = price_series.iloc[i - self.theta_window:i]
                try:
                    ou = OUEstimator()
                    ou.fit(window, dt=1.0)
                    theta_values.append(ou.theta_)
                except:
                    theta_values.append(np.nan)
        
        return pd.Series(theta_values, index=price_series.index)
    
    def _compute_rolling_entropy(self, price_series: pd.Series) -> pd.Series:
        """Compute rolling sample entropy on returns."""
        returns = price_series.pct_change()
        
        def entropy_func(window_values: np.ndarray) -> float:
            try:
                return calculate_shannon_entropy(window_values, bins=50)
            except:
                return np.nan
        
        return returns.rolling(
            window=self.entropy_window,
            min_periods=self.entropy_window
        ).apply(entropy_func, raw=True)
    
    def _compute_fracdiff_correlation(self, price_series: pd.Series) -> pd.Series:
        """
        Compute correlation between original and fractionally differentiated series.
        
        High correlation = strong memory retention even after stationarization.
        """
        # Find optimal d
        frac_diff = FractionalDifferentiator(window_size=self.fracdiff_window)
        
        n_calib = min(500, int(len(price_series) * 0.1))
        calib_series = price_series.iloc[:n_calib]
        
        try:
            self.optimal_d_ = frac_diff.find_min_d(calib_series, precision=0.05, verbose=False)
        except:
            self.optimal_d_ = 0.5
        
        # Apply fractional differentiation
        fracdiff_series = frac_diff.transform(price_series, d=self.optimal_d_)
        
        # Compute rolling correlation
        correlation = price_series.rolling(
            window=self.fracdiff_window,
            min_periods=self.fracdiff_window
        ).corr(fracdiff_series)
        
        return correlation
    
    def _compute_vol_of_vol(self, price_series: pd.Series) -> pd.Series:
        """
        Compute volatility of volatility (realized vol of realized vol).
        
        High vol-of-vol = explosive, regime-shifting behavior.
        """
        # Compute returns
        returns = price_series.pct_change()
        
        # Compute rolling volatility
        rolling_vol = returns.rolling(window=self.vol_window).std()
        
        # Compute volatility of volatility
        vol_of_vol = rolling_vol.rolling(window=self.vol_window).std()
        
        return vol_of_vol
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary table of all metrics.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for all metrics
        """
        if not self.results_:
            raise ValueError("Profiler has not been fitted yet. Call fit() first.")
        
        return pd.DataFrame(self.results_).T
    
    def diagnose(self) -> Dict[str, str]:
        """
        Provide automated diagnostic conclusions based on metric values.
        
        Returns
        -------
        Dict[str, str]
            Diagnostic conclusions for each metric
        """
        if not self.results_:
            raise ValueError("Profiler has not been fitted yet. Call fit() first.")
        
        diagnostics = {}
        
        # Hurst Exponent
        hurst_mean = self.results_['hurst']['mean']
        if hurst_mean > 0.55:
            diagnostics['hurst'] = f"TRENDING (H={hurst_mean:.3f}): Strong trend persistence. Trend Expert should work well."
        elif hurst_mean < 0.45:
            diagnostics['hurst'] = f"MEAN-REVERTING (H={hurst_mean:.3f}): Anti-persistent behavior. Range Expert preferred."
        else:
            diagnostics['hurst'] = f"RANDOM WALK (H={hurst_mean:.3f}): Near-random behavior. Low predictability."
        
        # Theta (Mean Reversion Speed)
        theta_mean = self.results_['theta']['mean']
        if theta_mean > 0.1:
            diagnostics['theta'] = f"ELASTIC (theta={theta_mean:.3f}): Fast mean reversion. Range strategies viable."
        elif theta_mean > 0.01:
            diagnostics['theta'] = f"MODERATE (theta={theta_mean:.3f}): Moderate mean reversion."
        else:
            diagnostics['theta'] = f"WEAK REVERSION (theta={theta_mean:.3f}): Slow/no mean reversion. Trend-following preferred."
        
        # Entropy
        entropy_mean = self.results_['entropy']['mean']
        if entropy_mean > 2.5:
            diagnostics['entropy'] = f"HIGH NOISE (S={entropy_mean:.3f}): High randomness. Difficult to predict."
        elif entropy_mean > 1.5:
            diagnostics['entropy'] = f"MODERATE NOISE (S={entropy_mean:.3f}): Moderate randomness."
        else:
            diagnostics['entropy'] = f"LOW NOISE (S={entropy_mean:.3f}): Low randomness. More predictable."
        
        # FracDiff Correlation
        fracdiff_mean = self.results_['fracdiff_corr']['mean']
        if fracdiff_mean > 0.7:
            diagnostics['fracdiff_corr'] = f"STRONG MEMORY (rho={fracdiff_mean:.3f}): High memory retention. Long-term patterns exist."
        elif fracdiff_mean > 0.3:
            diagnostics['fracdiff_corr'] = f"MODERATE MEMORY (rho={fracdiff_mean:.3f}): Some memory retention."
        else:
            diagnostics['fracdiff_corr'] = f"WEAK MEMORY (rho={fracdiff_mean:.3f}): Low memory. Mostly memoryless process."
        
        # Vol of Vol
        vov_mean = self.results_['vol_of_vol']['mean']
        vov_std = self.results_['vol_of_vol']['std']
        if vov_std / vov_mean > 1.0 if vov_mean > 0 else False:
            diagnostics['vol_of_vol'] = f"EXPLOSIVE (sigma_sigma={vov_mean:.6f}): Highly unstable volatility. Regime shifts common."
        else:
            diagnostics['vol_of_vol'] = f"STABLE (sigma_sigma={vov_mean:.6f}): Relatively stable volatility regime."
        
        return diagnostics
    
    def compare_with(self, other: PhysicsProfiler) -> pd.DataFrame:
        """
        Compare this profiler's results with another asset's profiler.
        
        Parameters
        ----------
        other : PhysicsProfiler
            Another fitted PhysicsProfiler instance
            
        Returns
        -------
        pd.DataFrame
            Comparison table showing differences
        """
        if not self.results_ or not other.results_:
            raise ValueError("Both profilers must be fitted before comparison.")
        
        comparison = []
        
        for metric in ['hurst', 'theta', 'entropy', 'fracdiff_corr', 'vol_of_vol']:
            self_mean = self.results_[metric]['mean']
            other_mean = other.results_[metric]['mean']
            diff = self_mean - other_mean
            pct_diff = (diff / other_mean * 100) if other_mean != 0 else np.nan
            
            comparison.append({
                'Metric': metric,
                'Asset_A_Mean': self_mean,
                'Asset_B_Mean': other_mean,
                'Difference': diff,
                'Pct_Difference': pct_diff,
            })
        
        return pd.DataFrame(comparison)


def compare_assets(
    asset_a_prices: pd.Series,
    asset_b_prices: pd.Series,
    asset_a_name: str = "Asset A",
    asset_b_name: str = "Asset B",
    verbose: bool = True,
) -> Tuple[PhysicsProfiler, PhysicsProfiler, pd.DataFrame]:
    """
    Convenience function to compare two assets.
    
    Parameters
    ----------
    asset_a_prices : pd.Series
        Price series for first asset
    asset_b_prices : pd.Series
        Price series for second asset
    asset_a_name : str
        Name of first asset (for display)
    asset_b_name : str
        Name of second asset (for display)
    verbose : bool
        Print progress and results
        
    Returns
    -------
    profiler_a : PhysicsProfiler
        Fitted profiler for asset A
    profiler_b : PhysicsProfiler
        Fitted profiler for asset B
    comparison : pd.DataFrame
        Comparison table
    """
    if verbose:
        print(f"\n{'=' * 72}")
        print(f"PHYSICS COMPARISON: {asset_a_name} vs {asset_b_name}")
        print(f"{'=' * 72}")
    
    # Profile Asset A
    if verbose:
        print(f"\n[1/2] Profiling {asset_a_name}...")
    profiler_a = PhysicsProfiler()
    profiler_a.fit(asset_a_prices, verbose=verbose)
    
    # Profile Asset B
    if verbose:
        print(f"\n[2/2] Profiling {asset_b_name}...")
    profiler_b = PhysicsProfiler()
    profiler_b.fit(asset_b_prices, verbose=verbose)
    
    # Compare
    comparison = profiler_a.compare_with(profiler_b)
    comparison.insert(1, 'Asset_A', asset_a_name)
    comparison.insert(3, 'Asset_B', asset_b_name)
    
    if verbose:
        print(f"\n{'=' * 72}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 72}")
        print(comparison.to_string(index=False))
        
        print(f"\n{'-' * 72}")
        print(f"{asset_a_name} DIAGNOSTICS")
        print(f"{'-' * 72}")
        for metric, diagnosis in profiler_a.diagnose().items():
            print(f"  {metric:15s}: {diagnosis}")
        
        print(f"\n{'-' * 72}")
        print(f"{asset_b_name} DIAGNOSTICS")
        print(f"{'-' * 72}")
        for metric, diagnosis in profiler_b.diagnose().items():
            print(f"  {metric:15s}: {diagnosis}")
    
    return profiler_a, profiler_b, comparison
