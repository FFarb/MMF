"""
Physics-Based Trading Experts.

This module implements trading experts based on physics and stochastic processes,
specifically designed to capture market behaviors that traditional ML models miss.

Experts:
1. OUMeanReversionExpert - Ornstein-Uhlenbeck process for mean reversion
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Optional


class OUMeanReversionExpert(BaseEstimator, ClassifierMixin):
    """
    Ornstein-Uhlenbeck Mean Reversion Expert.
    
    Physics-based expert that models price as an elastic process that reverts
    to equilibrium. Excels in choppy, sideways markets where traditional
    trend models fail.
    
    Theory:
    -------
    The Ornstein-Uhlenbeck process models mean reversion:
        dX_t = θ(μ - X_t)dt + σdW_t
    
    Where:
    - θ (theta): Mean reversion speed
    - μ (mu): Long-term equilibrium level
    - σ (sigma): Volatility
    
    Trading Logic:
    -------------
    - High Z-score (overbought) → Expect reversion down
    - Low Z-score (oversold) → Expect reversion up
    - Z-score near 0 → No strong signal
    
    Parameters
    ----------
    alpha : float, default=1.0
        Sensitivity scaling factor for sigmoid conversion
    lookback_window : int, default=100
        Window size for parameter calibration
    z_threshold : float, default=2.0
        Z-score threshold for strong signals
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        lookback_window: int = 100,
        z_threshold: float = 2.0,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.z_threshold = z_threshold
        self.random_state = random_state
        
        # OU parameters (fitted)
        self.theta_ = None
        self.mu_ = None
        self.sigma_ = None
        self.sigma_eq_ = None  # Equilibrium volatility
        
        self._fitted = False
        self.classes_ = np.array([0, 1])
    
    def _calibrate_ou_parameters(self, series: np.ndarray) -> tuple:
        """
        Calibrate Ornstein-Uhlenbeck parameters from time series.
        
        Uses discrete approximation:
        - μ (mu): Sample mean
        - θ (theta): Mean reversion speed from AR(1) coefficient
        - σ (sigma): Volatility from residuals
        
        Parameters
        ----------
        series : ndarray
            Time series data
        
        Returns
        -------
        theta, mu, sigma, sigma_eq : tuple
            Calibrated OU parameters
        """
        # Remove NaNs
        series = series[~np.isnan(series)]
        
        if len(series) < 10:
            # Not enough data, use defaults
            return 0.1, np.mean(series), np.std(series), np.std(series)
        
        # Estimate μ (equilibrium level) as sample mean
        mu = np.mean(series)
        
        # Estimate θ (mean reversion speed) using AR(1) regression
        # X_t = μ + ρ(X_{t-1} - μ) + ε_t
        # where ρ = exp(-θ * Δt), assuming Δt = 1
        
        X_t = series[1:]
        X_t_1 = series[:-1]
        
        # Demean
        X_t_demean = X_t - mu
        X_t_1_demean = X_t_1 - mu
        
        # AR(1) coefficient
        if np.std(X_t_1_demean) > 1e-8:
            rho = np.corrcoef(X_t_1_demean, X_t_demean)[0, 1]
            rho = np.clip(rho, -0.99, 0.99)  # Stability
        else:
            rho = 0.0
        
        # Mean reversion speed
        # ρ = exp(-θ), so θ = -log(ρ)
        if rho > 0:
            theta = -np.log(rho)
        else:
            theta = 0.1  # Default weak mean reversion
        
        theta = np.clip(theta, 0.01, 10.0)  # Reasonable bounds
        
        # Estimate σ (volatility) from residuals
        residuals = X_t_demean - rho * X_t_1_demean
        sigma = np.std(residuals)
        
        # Equilibrium volatility: σ_eq = σ / sqrt(2θ)
        sigma_eq = sigma / np.sqrt(2 * theta) if theta > 0 else sigma
        
        return theta, mu, sigma, sigma_eq
    
    def fit(self, X, y=None, sample_weight=None) -> "OUMeanReversionExpert":
        """
        Fit the OU mean reversion expert.
        
        Calibrates OU parameters (θ, μ, σ) from the time series.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix (expects 'frac_diff' or 'close' column)
        y : array, optional
            Target labels (not used, kept for sklearn compatibility)
        sample_weight : array, optional
            Sample weights (not used)
        
        Returns
        -------
        self : OUMeanReversionExpert
            Fitted expert
        """
        # Extract time series
        if isinstance(X, pd.DataFrame):
            # Prefer frac_diff, fallback to close
            if 'frac_diff' in X.columns:
                series = X['frac_diff'].values
            elif 'close' in X.columns:
                series = X['close'].values
            else:
                # Use first numeric column
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    series = X[numeric_cols[0]].values
                else:
                    raise ValueError("No numeric columns found in X")
        else:
            # Assume array-like
            series = np.asarray(X)
            if series.ndim > 1:
                series = series[:, 0]  # Take first column
        
        # Calibrate OU parameters
        self.theta_, self.mu_, self.sigma_, self.sigma_eq_ = self._calibrate_ou_parameters(series)
        
        self._fitted = True
        
        return self
    
    def _compute_z_score(self, series: np.ndarray) -> np.ndarray:
        """
        Compute Z-score relative to OU equilibrium.
        
        Z = (X_t - μ) / σ_eq
        
        Parameters
        ----------
        series : ndarray
            Time series data
        
        Returns
        -------
        z_scores : ndarray
            Z-scores for each point
        """
        if self.mu_ is None or self.sigma_eq_ is None:
            raise RuntimeError("Expert not fitted")
        
        z_scores = (series - self.mu_) / (self.sigma_eq_ + 1e-8)
        return z_scores
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities using OU mean reversion logic.
        
        Logic:
        ------
        - High Z-score (overbought) → Low P(Up) (expect reversion down)
        - Low Z-score (oversold) → High P(Up) (expect reversion up)
        
        Formula: P(Up) = 1 / (1 + exp(α * z))
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities [P(Down), P(Up)]
        """
        if not self._fitted:
            raise RuntimeError("OUMeanReversionExpert not fitted")
        
        # Extract time series
        if isinstance(X, pd.DataFrame):
            if 'frac_diff' in X.columns:
                series = X['frac_diff'].values
            elif 'close' in X.columns:
                series = X['close'].values
            else:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    series = X[numeric_cols[0]].values
                else:
                    raise ValueError("No numeric columns found in X")
        else:
            series = np.asarray(X)
            if series.ndim > 1:
                series = series[:, 0]
        
        # Compute Z-scores
        z_scores = self._compute_z_score(series)
        
        # Convert Z-scores to probabilities using inverted sigmoid
        # High Z (overbought) → Low P(Up)
        # Low Z (oversold) → High P(Up)
        p_up = 1.0 / (1.0 + np.exp(self.alpha * z_scores))
        
        # Clip to reasonable range
        p_up = np.clip(p_up, 0.01, 0.99)
        
        # Create probability matrix [P(Down), P(Up)]
        proba = np.column_stack([1.0 - p_up, p_up])
        
        return proba
    
    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        
        Returns
        -------
        predictions : ndarray
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_ou_parameters(self) -> dict:
        """
        Get calibrated OU parameters.
        
        Returns
        -------
        params : dict
            Dictionary with theta, mu, sigma, sigma_eq
        """
        if not self._fitted:
            return {}
        
        return {
            "theta": float(self.theta_),
            "mu": float(self.mu_),
            "sigma": float(self.sigma_),
            "sigma_eq": float(self.sigma_eq_),
            "half_life": float(np.log(2) / self.theta_) if self.theta_ > 0 else np.inf,
        }
