"""
Physics-Based Trading Experts.

This module implements trading experts based on physics and stochastic processes,
specifically designed to capture market behaviors that traditional ML models miss.

Experts:
1. OUMeanReversionExpert - Ornstein-Uhlenbeck process for mean reversion
2. NeuralODEExpert - Neural Ordinary Differential Equations for non-linear dynamics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
    use_volatility_filter : bool, default=True
        Whether to filter signals based on volatility regime
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        lookback_window: int = 100,
        z_threshold: float = 2.0,
        use_volatility_filter: bool = True,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.z_threshold = z_threshold
        self.use_volatility_filter = use_volatility_filter
        self.random_state = random_state
        
        # Fitted parameters
        self.theta_ = None  # Mean reversion speed
        self.mu_ = None     # Equilibrium level
        self.sigma_ = None  # Volatility
        self.sigma_eq_ = None  # Equilibrium volatility
        self._fitted = False
    
    def _calibrate_ou_params(self, prices: np.ndarray) -> tuple:
        """
        Calibrate Ornstein-Uhlenbeck parameters from price series.
        
        Uses Maximum Likelihood Estimation for discrete observations.
        
        Parameters
        ----------
        prices : np.ndarray
            Price time series
        
        Returns
        -------
        theta, mu, sigma : tuple
            Calibrated OU parameters
        """
        # Calculate log returns
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        
        # Estimate parameters
        # θ (mean reversion speed)
        # Approximate using autocorrelation
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            theta = -np.log(max(abs(autocorr), 0.01))  # Avoid log(0)
        else:
            theta = 0.1  # Default
        
        # μ (equilibrium level)
        mu = np.mean(log_prices)
        
        # σ (volatility)
        sigma = np.std(returns) * np.sqrt(252)  # Annualized
        
        return theta, mu, sigma
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight=None):
        """
        Fit the OU mean reversion model.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix (must contain 'close' or 'frac_diff')
        y : array
            Target labels (not used, but required for sklearn compatibility)
        sample_weight : array, optional
            Sample weights
        
        Returns
        -------
        self
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Extract price series
        if 'close' in X.columns:
            prices = X['close'].values
        elif 'frac_diff' in X.columns:
            # Use cumulative sum of frac_diff as proxy for price
            prices = np.exp(np.cumsum(X['frac_diff'].values))
        else:
            # Fallback: use first column
            prices = X.iloc[:, 0].values
        
        # Calibrate parameters
        self.theta_, self.mu_, self.sigma_ = self._calibrate_ou_params(prices)
        
        # Calculate equilibrium volatility (for filtering)
        self.sigma_eq_ = np.std(prices[-self.lookback_window:]) if len(prices) >= self.lookback_window else np.std(prices)
        
        self._fitted = True
        return self
    
    def _calculate_ou_zscore(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate OU Z-score for each sample.
        
        Z-score = (X - μ) / σ_eq
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix
        
        Returns
        -------
        z_scores : np.ndarray
            OU Z-scores
        """
        # Extract price series
        if 'close' in X.columns:
            prices = X['close'].values
        elif 'frac_diff' in X.columns:
            prices = np.exp(np.cumsum(X['frac_diff'].values))
        else:
            prices = X.iloc[:, 0].values
        
        # Calculate Z-score
        log_prices = np.log(prices)
        z_scores = (log_prices - self.mu_) / (self.sigma_eq_ + 1e-8)
        
        return z_scores
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using OU mean reversion.
        
        Logic:
        ------
        - Negative Z-score (oversold) → Higher probability of UP (class 1)
        - Positive Z-score (overbought) → Lower probability of UP (class 0)
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        
        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Class probabilities [P(down), P(up)]
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Calculate Z-scores
        z_scores = self._calculate_ou_zscore(X)
        
        # Convert Z-score to probability
        # Negative Z → High P(up), Positive Z → Low P(up)
        # Use sigmoid: P(up) = 1 / (1 + exp(alpha * z))
        p_up = 1.0 / (1.0 + np.exp(self.alpha * z_scores))
        
        # Apply volatility filter if enabled
        if self.use_volatility_filter and 'volatility' in X.columns:
            vol = X['volatility'].values
            vol_threshold = np.median(vol)
            
            # Reduce confidence in high volatility
            high_vol_mask = vol > vol_threshold
            p_up[high_vol_mask] = 0.5 + (p_up[high_vol_mask] - 0.5) * 0.5
        
        # Clip to valid probability range
        p_up = np.clip(p_up, 0.01, 0.99)
        p_down = 1.0 - p_up
        
        return np.column_stack([p_down, p_up])
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        
        Returns
        -------
        predictions : np.ndarray
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'alpha': self.alpha,
            'lookback_window': self.lookback_window,
            'z_threshold': self.z_threshold,
            'use_volatility_filter': self.use_volatility_filter,
            'random_state': self.random_state,
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_model_params(self) -> dict:
        """
        Get fitted OU model parameters.
        
        Returns
        -------
        params : dict
            Dictionary containing theta, mu, sigma, and half-life
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


# ==============================================================================
# NEURAL ODE EXPERT
# ==============================================================================

class ODEF(nn.Module):
    """
    Ordinary Differential Equation Function (Neural Network).
    
    Represents the derivative function f(h, t) in the ODE:
        dh/dt = f(h, t)
    
    This is a small MLP that learns the dynamics of the latent state.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim),
            nn.Tanh()
        )
    
    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt = f(h, t).
        
        Parameters
        ----------
        t : float
            Time (not used in autonomous ODE, but kept for compatibility)
        h : torch.Tensor
            Current state
        
        Returns
        -------
        dhdt : torch.Tensor
            Derivative of state
        """
        return self.net(h)


class NeuralODEExpert(BaseEstimator, ClassifierMixin):
    """
    Neural Ordinary Differential Equation Expert.
    
    Models market state evolution as a continuous dynamical system:
        dh/dt = f_θ(h, t)
    
    Where:
    - h: Latent market state
    - f_θ: Neural network (learned)
    - t: Time
    
    This captures non-linear dynamics that linear models (like OU) miss,
    making it superior for assets with complex behavior (DOGE, AVAX).
    
    Architecture:
    -------------
    Input → Encoder → ODE Solver → Decoder → Probability
    
    The ODE solver evolves the latent state forward in time,
    then the decoder maps it to a trading signal.
    
    Parameters
    ----------
    input_dim : int, optional
        Input feature dimension (auto-detected if None)
    hidden_dim : int, default=32
        Latent state dimension
    lr : float, default=0.01
        Learning rate
    epochs : int, default=50
        Training epochs
    time_steps : int, default=10
        Number of ODE integration steps
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 32,
        lr: float = 0.01,
        epochs: int = 50,
        time_steps: int = 10,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.time_steps = time_steps
        self.random_state = random_state
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.ode_func = None
        self.decoder = None
        self._fitted = False
    
    def _build_model(self, input_dim: int):
        """Build the Neural ODE model."""
        # Encoder: Input → Latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh()
        ).to(self.device)
        
        # ODE Function: dh/dt = f(h)
        self.ode_func = ODEF(self.hidden_dim).to(self.device)
        
        # Decoder: Latent → Probability
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
    
    def _ode_step(self, h: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Single Euler step for ODE integration.
        
        h_{t+1} = h_t + dt * f(h_t, t)
        
        Parameters
        ----------
        h : torch.Tensor
            Current state
        dt : float
            Time step
        
        Returns
        -------
        h_next : torch.Tensor
            Next state
        """
        dhdt = self.ode_func(0.0, h)  # Autonomous ODE (t not used)
        return h + dt * dhdt
    
    def _integrate_ode(self, h0: torch.Tensor) -> torch.Tensor:
        """
        Integrate ODE forward in time.
        
        Parameters
        ----------
        h0 : torch.Tensor
            Initial state
        
        Returns
        -------
        h_final : torch.Tensor
            Final state after integration
        """
        h = h0
        dt = 1.0 / self.time_steps
        
        for _ in range(self.time_steps):
            h = self._ode_step(h, dt)
        
        return h
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight=None):
        """
        Train the Neural ODE model.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        y : array
            Target labels (0 or 1)
        sample_weight : array, optional
            Sample weights (not used)
        
        Returns
        -------
        self
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_np = X.select_dtypes(include=[np.number]).values
        else:
            X_np = X
        
        # Auto-detect input dimension
        if self.input_dim is None:
            self.input_dim = X_np.shape[1]
        
        # Build model
        self._build_model(self.input_dim)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_np).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        # Optimizer and loss
        params = list(self.encoder.parameters()) + \
                 list(self.ode_func.parameters()) + \
                 list(self.decoder.parameters())
        optimizer = optim.Adam(params, lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.encoder.train()
        self.ode_func.train()
        self.decoder.train()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            h0 = self.encoder(X_tensor)
            h_final = self._integrate_ode(h0)
            logits = self.decoder(h_final)
            
            # Loss
            loss = criterion(logits, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"  [NeuralODE] Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self._fitted = True
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        
        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Class probabilities [P(down), P(up)]
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_np = X.select_dtypes(include=[np.number]).values
        else:
            X_np = X
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_np).to(self.device)
        
        # Inference mode
        self.encoder.eval()
        self.ode_func.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Forward pass
            h0 = self.encoder(X_tensor)
            h_final = self._integrate_ode(h0)
            logits = self.decoder(h_final)
            
            # Convert to probability
            p_up = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # Clip to valid range
        p_up = np.clip(p_up, 0.01, 0.99)
        p_down = 1.0 - p_up
        
        return np.column_stack([p_down, p_up])
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        
        Returns
        -------
        predictions : np.ndarray
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'epochs': self.epochs,
            'time_steps': self.time_steps,
            'random_state': self.random_state,
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
