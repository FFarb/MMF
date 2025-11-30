"""
Nonlinear Dynamics Engine - Double-Well Potential Model

This module implements a nonlinear extension to the OU baseline using a
Double-Well potential (Landau-Ginzburg model) to capture market regime bistability.

The underlying SDE is:
    dX_t = -dV(X)/dX dt + sigma dW_t

Where the potential is:
    V(x) = -a/2 * x^2 + b/4 * x^4  (Double-Well if a > 0, b > 0)

Drift term:
    f(x) = ax - bx^3

This captures bistability: the system has two stable equilibria (regimes)
separated by an unstable equilibrium (tipping point).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class DoubleWellEstimator:
    """
    Double-Well potential estimator for nonlinear market dynamics.
    
    Fits a stochastic process with cubic drift to capture regime bistability.
    The model can exhibit either monostable (single regime) or bistable
    (two regime) behavior depending on the fitted parameters.
    
    Attributes
    ----------
    a_ : float
        Linear coefficient (fitted)
    b_ : float
        Cubic coefficient (fitted)
    sigma_ : float
        Volatility (fitted)
    is_bistable_ : bool
        True if a > 0 and b > 0 (bistable system)
    tipping_points_ : np.ndarray
        Equilibrium points (roots of drift equation)
    mean_ : float
        Original series mean (for unstandardization)
    std_ : float
        Original series std (for unstandardization)
    r_squared_ : float
        R^2 of the polynomial fit
    """
    
    def __init__(self) -> None:
        self.a_: float = 0.0
        self.b_: float = 0.0
        self.sigma_: float = 0.0
        self.is_bistable_: bool = False
        self.tipping_points_: np.ndarray = np.array([])
        self.mean_: float = 0.0
        self.std_: float = 1.0
        self.r_squared_: float = 0.0
        
    def fit(self, series: pd.Series, dt: float = 1.0) -> DoubleWellEstimator:
        """
        Fit the Double-Well model to a discrete time series.
        
        Standardizes the data, fits a cubic polynomial to the drift,
        and extracts the linear and cubic coefficients.
        
        Parameters
        ----------
        series : pd.Series
            Time series data to fit
        dt : float, default=1.0
            Time step between observations
            
        Returns
        -------
        self : DoubleWellEstimator
            Fitted estimator instance
            
        Notes
        -----
        The drift is approximated as:
            dX/dt ≈ a*X + b*X^3
        
        Bistability occurs when a > 0 and b > 0, creating a W-shaped potential.
        """
        # Handle NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            raise ValueError(f"Insufficient data points: {len(clean_series)} (minimum 10 required)")
        
        # Standardize the series (zero mean, unit variance)
        self.mean_ = float(clean_series.mean())
        self.std_ = float(clean_series.std())
        
        if self.std_ < 1e-10:
            raise ValueError("Series has zero variance, cannot fit model")
        
        standardized = (clean_series - self.mean_) / self.std_
        
        # Prepare data for polynomial regression
        X_t = standardized.values[:-1]
        X_t_plus_1 = standardized.values[1:]
        
        # Calculate discrete increments
        delta_X = X_t_plus_1 - X_t
        
        # Create polynomial features: [X, X^3]
        # We want only linear and cubic terms, no bias, no quadratic
        X_features = np.column_stack([X_t, X_t ** 3])
        
        # Fit linear regression: dX/dt ≈ a*X + b*X^3
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_features, delta_X / dt)
        
        # Extract coefficients
        self.a_ = float(reg.coef_[0])
        self.b_ = float(reg.coef_[1])
        
        # Calculate residuals
        predictions = reg.predict(X_features) * dt
        residuals = delta_X - predictions
        
        # Estimate sigma from residuals
        self.sigma_ = float(np.std(residuals, ddof=1) / np.sqrt(dt))
        
        # Calculate R^2
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((delta_X - np.mean(delta_X)) ** 2)
        self.r_squared_ = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Determine bistability
        self.is_bistable_ = (self.a_ > 0) and (self.b_ > 0)
        
        # Calculate tipping points (equilibria): solve ax - bx^3 = 0
        # x(a - bx^2) = 0 => x = 0 or x = ±sqrt(a/b)
        if self.is_bistable_:
            x_star = np.sqrt(self.a_ / self.b_)
            self.tipping_points_ = np.array([-x_star, 0.0, x_star])
        elif self.a_ < 0:
            # Monostable at x = 0
            self.tipping_points_ = np.array([0.0])
        else:
            # Edge case: a > 0 but b <= 0 (unstable)
            self.tipping_points_ = np.array([0.0])
        
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
        Generate synthetic paths using the fitted Double-Well parameters.
        
        Uses Euler-Maruyama discretization with numerical stability handling.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate
        n_paths : int, default=1
            Number of independent paths to generate
        initial_value : float, optional
            Starting value for the paths. If None, uses 0 (standardized mean)
        dt : float, default=1.0
            Time step size
        random_state : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        paths : np.ndarray
            Simulated paths of shape (n_steps, n_paths)
            Values are in standardized space (zero mean, unit variance)
            
        Notes
        -----
        Euler-Maruyama scheme:
            X_{t+1} = X_t + (a*X_t - b*X_t^3)*dt + sigma*sqrt(dt)*Z
        
        Numerical stability: clamps extreme values to prevent explosion.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize paths (in standardized space)
        paths = np.zeros((n_steps, n_paths))
        x0 = initial_value if initial_value is not None else 0.0
        paths[0, :] = x0
        
        # Generate random shocks
        sqrt_dt = np.sqrt(dt)
        dW = np.random.randn(n_steps - 1, n_paths) * sqrt_dt
        
        # Euler-Maruyama scheme with stability handling
        max_value = 10.0  # Clamp threshold
        
        for t in range(1, n_steps):
            X_t = paths[t - 1, :]
            
            # Drift: f(x) = a*x - b*x^3
            drift = (self.a_ * X_t - self.b_ * X_t ** 3) * dt
            
            # Diffusion
            diffusion = self.sigma_ * dW[t - 1, :]
            
            # Update
            paths[t, :] = X_t + drift + diffusion
            
            # Numerical stability: clamp extreme values
            paths[t, :] = np.clip(paths[t, :], -max_value, max_value)
        
        return paths
    
    def get_potential_shape(
        self,
        x_range: tuple[float, float] = (-3.0, 3.0),
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the potential landscape V(x) for visualization.
        
        Parameters
        ----------
        x_range : tuple of float, default=(-3, 3)
            Range of x values to evaluate
        n_points : int, default=200
            Number of points to evaluate
            
        Returns
        -------
        x : np.ndarray
            X coordinates
        V : np.ndarray
            Potential values V(x) = -a/2*x^2 + b/4*x^4
            
        Notes
        -----
        The shape of V(x) indicates the regime structure:
        - U-shape (parabola): Monostable, single equilibrium
        - W-shape (double well): Bistable, two stable regimes
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        # V(x) = -a/2 * x^2 + b/4 * x^4
        V = -0.5 * self.a_ * x ** 2 + 0.25 * self.b_ * x ** 4
        
        return x, V
    
    def diagnostics(self) -> dict:
        """
        Return diagnostic information about the fitted model.
        
        Returns
        -------
        dict
            Dictionary containing:
            - a: Linear coefficient
            - b: Cubic coefficient
            - sigma: Volatility
            - is_bistable: Boolean indicating bistability
            - tipping_points: Equilibrium points
            - potential_type: Description of potential shape
            - r_squared: R^2 of the fit
        """
        if self.is_bistable_:
            potential_type = "Bistable W-shape (two regimes)"
        elif self.a_ < 0:
            potential_type = "Monostable U-shape (single regime)"
        else:
            potential_type = "Unstable (a > 0, b <= 0)"
        
        return {
            "a": self.a_,
            "b": self.b_,
            "sigma": self.sigma_,
            "is_bistable": self.is_bistable_,
            "tipping_points": self.tipping_points_.tolist(),
            "potential_type": potential_type,
            "r_squared": self.r_squared_,
            "mean": self.mean_,
            "std": self.std_,
        }
