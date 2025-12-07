"""
Ornstein-Uhlenbeck Process Calibration Engine

This module provides a mathematically rigorous calibration engine for the
Ornstein-Uhlenbeck (OU) process, mapping discrete market data to continuous-time
OU parameters using exact discretization formulas.

The OU process is defined by the SDE:
    dX_t = θ(μ - X_t)dt + σdW_t

Where:
    θ (theta): Mean reversion speed
    μ (mu): Long-term mean
    σ (sigma): Volatility of the process
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class OUEstimator:
    """
    Ornstein-Uhlenbeck process parameter estimator.
    
    Fits an OU process to discrete time series data using exact discretization
    formulas to map regression coefficients to physical OU parameters.
    
    Attributes
    ----------
    theta_ : float
        Mean reversion speed (fitted)
    mu_ : float
        Long-term mean (fitted)
    sigma_ : float
        Volatility (fitted)
    half_life_ : float
        Half-life of mean reversion in time units
    alpha_ : float
        Regression intercept
    beta_ : float
        Regression slope
    residuals_ : np.ndarray
        Regression residuals
    r_squared_ : float
        R² of the linear regression fit
    """
    
    def __init__(self) -> None:
        self.theta_: float = 0.0
        self.mu_: float = 0.0
        self.sigma_: float = 0.0
        self.half_life_: float = np.inf
        self.alpha_: float = 0.0
        self.beta_: float = 0.0
        self.residuals_: np.ndarray = np.array([])
        self.r_squared_: float = 0.0
        
    def fit(self, series: pd.Series, dt: float = 1.0) -> OUEstimator:
        """
        Fit the OU process to a discrete time series.
        
        Performs linear regression X_{t+1} = α + β X_t + ε and maps the
        coefficients to OU parameters using exact discretization formulas.
        
        Parameters
        ----------
        series : pd.Series
            Time series data to fit
        dt : float, default=1.0
            Time step between observations
            
        Returns
        -------
        self : OUEstimator
            Fitted estimator instance
            
        Notes
        -----
        The mapping formulas are:
            θ = -ln(β) / dt
            μ = α / (1 - β)
            σ = std(ε) × sqrt(-2ln(β) / (dt(1 - β²)))
            t_{1/2} = ln(2) / θ
            
        Edge cases:
            - If β >= 1: Process is random walk or explosive, set θ = 0
            - NaN values are dropped before regression
        """
        # Handle NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            raise ValueError(f"Insufficient data points: {len(clean_series)} (minimum 10 required)")
        
        # Prepare regression data: X_t and X_{t+1}
        X_t = clean_series.values[:-1].reshape(-1, 1)
        X_t_plus_1 = clean_series.values[1:]
        
        # Perform linear regression: X_{t+1} = α + β X_t + ε
        reg = LinearRegression()
        reg.fit(X_t, X_t_plus_1)
        
        self.alpha_ = float(reg.intercept_)
        self.beta_ = float(reg.coef_[0])
        
        # Calculate residuals
        predictions = reg.predict(X_t)
        self.residuals_ = X_t_plus_1 - predictions
        
        # Calculate R²
        ss_res = np.sum(self.residuals_ ** 2)
        ss_tot = np.sum((X_t_plus_1 - np.mean(X_t_plus_1)) ** 2)
        self.r_squared_ = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Map to OU parameters using exact discretization formulas
        if self.beta_ >= 1.0:
            # Random walk or explosive process
            self.theta_ = 0.0
            self.mu_ = np.mean(clean_series)
            self.sigma_ = np.std(self.residuals_)
            self.half_life_ = np.inf
        else:
            # Mean-reverting process
            # θ = -ln(β) / dt
            self.theta_ = -np.log(self.beta_) / dt
            
            # μ = α / (1 - β)
            self.mu_ = self.alpha_ / (1.0 - self.beta_)
            
            # σ = std(ε) × sqrt(-2ln(β) / (dt(1 - β²)))
            residual_std = np.std(self.residuals_, ddof=1)
            beta_squared = self.beta_ ** 2
            
            if beta_squared < 1.0:  # Additional safety check
                sigma_multiplier = np.sqrt(-2.0 * np.log(self.beta_) / (dt * (1.0 - beta_squared)))
                self.sigma_ = residual_std * sigma_multiplier
            else:
                self.sigma_ = residual_std
            
            # Half-life: t_{1/2} = ln(2) / θ
            if self.theta_ > 0:
                self.half_life_ = np.log(2.0) / self.theta_
            else:
                self.half_life_ = np.inf
        
        return self
    
    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        initial_value: Optional[float] = None,
        dt: float = 1.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate synthetic paths using the fitted OU parameters.
        
        Uses the Euler-Maruyama discretization scheme for simulation.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate
        n_paths : int, default=1
            Number of independent paths to generate
        initial_value : float, optional
            Starting value for the paths. If None, uses mu_
        dt : float, default=1.0
            Time step size
        random_state : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        paths : np.ndarray
            Simulated paths of shape (n_steps, n_paths)
            
        Notes
        -----
        Euler-Maruyama discretization:
            X_{t+dt} = X_t + θ(μ - X_t)dt + σ√dt × Z
        where Z ~ N(0, 1)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize paths
        paths = np.zeros((n_steps, n_paths))
        x0 = initial_value if initial_value is not None else self.mu_
        paths[0, :] = x0
        
        # Generate random shocks
        sqrt_dt = np.sqrt(dt)
        dW = np.random.randn(n_steps - 1, n_paths) * sqrt_dt
        
        # Euler-Maruyama scheme
        for t in range(1, n_steps):
            drift = self.theta_ * (self.mu_ - paths[t - 1, :]) * dt
            diffusion = self.sigma_ * dW[t - 1, :]
            paths[t, :] = paths[t - 1, :] + drift + diffusion
        
        return paths
    
    def diagnostics(self) -> dict:
        """
        Return diagnostic information about the fitted model.
        
        Returns
        -------
        dict
            Dictionary containing:
            - theta: Mean reversion speed
            - mu: Long-term mean
            - sigma: Volatility
            - half_life: Half-life of mean reversion
            - alpha: Regression intercept
            - beta: Regression slope
            - r_squared: R² of the fit
            - residuals_std: Standard deviation of residuals
            - is_mean_reverting: Boolean indicating if θ > 0
        """
        return {
            "theta": self.theta_,
            "mu": self.mu_,
            "sigma": self.sigma_,
            "half_life": self.half_life_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "r_squared": self.r_squared_,
            "residuals_std": float(np.std(self.residuals_, ddof=1)) if len(self.residuals_) > 0 else 0.0,
            "is_mean_reverting": self.theta_ > 0,
        }
