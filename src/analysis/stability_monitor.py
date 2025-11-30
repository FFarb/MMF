"""
Stability Monitor - Early Warning Signals for Critical Slowing Down

This module implements rolling window analysis to detect Critical Slowing Down (CSD)
and Early Warning Signals (EWS) before regime transitions and crashes.

Theory:
Before a critical transition, systems exhibit:
- Increased autocorrelation (sluggish recovery)
- Increased variance (flickering between states)
- Decreased mean reversion speed (theta -> 0)

These are universal early warning signals from chaos theory and complex systems.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class StabilityMonitor:
    """
    Rolling window stability monitor for detecting Early Warning Signals.
    
    Computes three key indicators of Critical Slowing Down:
    1. Lag-1 Autocorrelation (ACF-1): Increases toward 1.0 before transitions
    2. Variance: Spikes due to flickering between states
    3. Dynamic Theta: Decreases toward 0 as mean reversion is lost
    """
    
    @staticmethod
    def compute_indicators(
        series: pd.Series,
        window_size: int = 100,
        dt: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute rolling window stability indicators.
        
        Parameters
        ----------
        series : pd.Series
            Time series data (e.g., log-prices)
        window_size : int, default=100
            Size of rolling window in data points
        dt : float, default=1.0
            Time step between observations
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['acf1', 'variance', 'theta']
            Index matches the input series (NaN for first window_size points)
            
        Notes
        -----
        For each rolling window:
        - ACF-1: Lag-1 autocorrelation, ρ₁ = corr(X_t, X_{t-1})
        - Variance: Var(X_t) within window
        - Theta: From OU fit X_{t+1} = α + β X_t, θ = -ln(β)/dt
        """
        n = len(series)
        
        # Initialize result arrays
        acf1 = np.full(n, np.nan)
        variance = np.full(n, np.nan)
        theta = np.full(n, np.nan)
        
        # Convert to numpy for speed
        values = series.values
        
        # Rolling window computation
        for i in range(window_size, n):
            window = values[i - window_size:i]
            
            # 1. Lag-1 Autocorrelation
            # Compute correlation between X_t and X_{t-1}
            x_t = window[1:]
            x_t_minus_1 = window[:-1]
            
            if len(x_t) > 1:
                # Pearson correlation
                corr_matrix = np.corrcoef(x_t, x_t_minus_1)
                acf1[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            else:
                acf1[i] = 0.0
            
            # 2. Variance
            variance[i] = np.var(window, ddof=1)
            
            # 3. Dynamic Theta (OU calibration)
            # Fit: X_{t+1} = α + β X_t
            X_t = window[:-1].reshape(-1, 1)
            X_t_plus_1 = window[1:]
            
            if len(X_t) > 2:
                # Simple linear regression (manual for speed)
                X_mean = np.mean(X_t)
                Y_mean = np.mean(X_t_plus_1)
                
                numerator = np.sum((X_t.flatten() - X_mean) * (X_t_plus_1 - Y_mean))
                denominator = np.sum((X_t.flatten() - X_mean) ** 2)
                
                if denominator > 1e-10:
                    beta = numerator / denominator
                    
                    # Calculate theta
                    if beta > 0 and beta < 1.0:
                        theta[i] = -np.log(beta) / dt
                    elif beta >= 1.0:
                        theta[i] = 0.0  # No mean reversion (critical)
                    else:
                        theta[i] = 0.0  # Negative beta (unstable)
                else:
                    theta[i] = 0.0
            else:
                theta[i] = 0.0
        
        # Create DataFrame
        result = pd.DataFrame({
            'acf1': acf1,
            'variance': variance,
            'theta': theta,
        }, index=series.index)
        
        return result
    
    @staticmethod
    def detect_warnings(
        indicators_df: pd.DataFrame,
        threshold_z: float = 2.0,
        z_window: int = 50,
    ) -> pd.DataFrame:
        """
        Detect early warning signals using Z-score thresholds.
        
        Parameters
        ----------
        indicators_df : pd.DataFrame
            Output from compute_indicators()
        threshold_z : float, default=2.0
            Z-score threshold for warnings (standard deviations)
        z_window : int, default=50
            Window size for computing rolling mean/std for Z-scores
            
        Returns
        -------
        pd.DataFrame
            Original indicators plus Z-scores and warning flags:
            - z_acf1, z_variance, z_theta: Z-scores
            - warning_acf: ACF rising (z > threshold)
            - warning_theta: Theta falling (z < -threshold)
            - warning_combined: Either warning is True
            
        Notes
        -----
        Z-score normalization:
            z = (x - rolling_mean) / rolling_std
        
        Warning conditions:
            - ACF Warning: z_acf > threshold (autocorrelation increasing)
            - Theta Warning: z_theta < -threshold (mean reversion decreasing)
        """
        result = indicators_df.copy()
        
        # Compute Z-scores using rolling statistics
        for col in ['acf1', 'variance', 'theta']:
            if col in result.columns:
                rolling_mean = result[col].rolling(window=z_window, min_periods=1).mean()
                rolling_std = result[col].rolling(window=z_window, min_periods=1).std()
                
                # Avoid division by zero
                rolling_std = rolling_std.replace(0, 1e-10)
                
                # Calculate Z-score
                result[f'z_{col}'] = (result[col] - rolling_mean) / rolling_std
        
        # Detect warnings
        result['warning_acf'] = result['z_acf1'] > threshold_z
        result['warning_theta'] = result['z_theta'] < -threshold_z
        result['warning_combined'] = result['warning_acf'] | result['warning_theta']
        
        return result
    
    @staticmethod
    def identify_crashes(
        prices: pd.Series,
        threshold_pct: float = 5.0,
        window_hours: int = 24,
    ) -> pd.Series:
        """
        Identify significant price crashes.
        
        Parameters
        ----------
        prices : pd.Series
            Price time series
        threshold_pct : float, default=5.0
            Minimum percentage drop to classify as crash
        window_hours : int, default=24
            Time window to measure drop
            
        Returns
        -------
        pd.Series
            Boolean series indicating crash points
        """
        # Calculate rolling maximum over past window
        rolling_max = prices.rolling(window=window_hours, min_periods=1).max()
        
        # Calculate drawdown from recent high
        drawdown_pct = ((prices - rolling_max) / rolling_max) * 100
        
        # Identify crashes (drops exceeding threshold)
        crashes = drawdown_pct < -threshold_pct
        
        return crashes
