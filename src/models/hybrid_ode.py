"""
Hybrid Neural ODE Expert with Physical Priors and Sparsity Constraints.

This implements a scientifically rigorous approach to Neural ODEs:
- Linear physical prior (mean reversion / harmonic oscillator)
- Sparse neural correction term
- Gating mechanism to balance physics vs learning
- Jacobian regularization for smooth dynamics
- "Neuron burning" to prevent overfitting

Architecture:
    dy/dt = A·y + α·NeuralNet(y, t)
    
Where:
- A·y: Linear physics (stable baseline)
- α: Learnable gate (prefers physics if neural doesn't help)
- NeuralNet: Sparse correction for non-linear effects

Loss:
    L = MSE + λ₁·L1(weights) + λ₂·L1(α) + λ₃·||∇_y f||²
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Optional
import warnings

# Try to import torchdiffeq, fallback to manual Euler if not available
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    warnings.warn("torchdiffeq not available, using manual Euler integration")


class HybridODEFunc(nn.Module):
    """
    Hybrid ODE Function: Linear Physics + Sparse Neural Correction.
    
    Dynamics:
        dy/dt = A·y + α·NeuralNet(y, t)
    
    Components:
    -----------
    A : Linear layer (no bias)
        Physical prior - mean reversion or harmonic oscillator
        Initialized to be stable (eigenvalues < 0)
    
    NeuralNet : Small MLP
        Sparse correction for non-linear effects
        2 hidden layers with Tanh activation
    
    α : Learnable scalar gate
        Controls neural influence
        Initialized near 0 (prefer physics)
        L1 penalty forces it to stay small unless neural helps
    """
    
    def __init__(self, dim: int, hidden_dim: int = 16):
        super().__init__()
        
        # Linear Physics Prior: A·y
        self.linear = nn.Linear(dim, dim, bias=False)
        
        # Initialize to stable mean reversion
        # A = -θ·I where θ > 0 (mean reversion speed)
        with torch.no_grad():
            self.linear.weight.copy_(-0.1 * torch.eye(dim))
        
        # Neural Correction: Small sparse MLP
        self.neural_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Gating mechanism: α (scalar)
        # Initialized near 0 to prefer physics
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
        self.dim = dim
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute dy/dt = A·y + α·NeuralNet(y, t).
        
        Parameters
        ----------
        t : torch.Tensor
            Time (scalar or batch)
        y : torch.Tensor
            State (batch_size, dim)
        
        Returns
        -------
        dydt : torch.Tensor
            Derivative (batch_size, dim)
        """
        # Linear physics term
        linear_term = self.linear(y)
        
        # Neural correction term
        # Concatenate state and time
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        t_expanded = t.expand(y.shape[0], 1)
        y_t = torch.cat([y, t_expanded], dim=-1)
        
        neural_term = self.neural_net(y_t)
        
        # Gated combination
        dydt = linear_term + self.alpha * neural_term
        
        return dydt.squeeze(0) if dydt.shape[0] == 1 else dydt
    
    def get_l1_penalty(self) -> torch.Tensor:
        """
        Compute L1 penalty on neural network weights.
        
        This enforces sparsity - "neuron burning".
        Only keep neurons that significantly improve predictions.
        """
        l1_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for param in self.neural_net.parameters():
            l1_loss += torch.sum(torch.abs(param))
        
        return l1_loss
    
    def get_gate_penalty(self) -> torch.Tensor:
        """
        Compute L1 penalty on gate parameter α.
        
        Forces model to prefer linear physics unless
        neural correction significantly improves predictions.
        """
        return torch.abs(self.alpha)
    
    def compute_jacobian_reg(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian regularization: ||∇_y f(y,t)||²_F.
        
        This penalizes chaotic/unstable dynamics.
        Forces learned dynamics to be smooth and predictable.
        
        Parameters
        ----------
        y : torch.Tensor
            State (batch_size, dim)
        t : torch.Tensor
            Time
        
        Returns
        -------
        jac_reg : torch.Tensor
            Frobenius norm of Jacobian
        """
        y = y.requires_grad_(True)
        
        # Compute f(y, t)
        dydt = self.forward(t, y)
        
        # Compute Jacobian ∇_y f
        jac_norm = torch.tensor(0.0, device=y.device)
        
        for i in range(self.dim):
            # Gradient of i-th output w.r.t. y
            grad_outputs = torch.zeros_like(dydt)
            grad_outputs[:, i] = 1.0
            
            grads = torch.autograd.grad(
                outputs=dydt,
                inputs=y,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Add to Frobenius norm
            jac_norm += torch.sum(grads ** 2)
        
        return jac_norm


class HybridNeuralODEExpert(BaseEstimator, ClassifierMixin):
    """
    Hybrid Neural ODE Expert with Physical Priors.
    
    Combines:
    - Linear mean reversion (stable baseline)
    - Sparse neural correction (non-linear effects)
    - Gating mechanism (automatic model selection)
    - Regularization (prevent overfitting)
    
    This is superior to pure Neural ODE because:
    1. Physical prior provides stable baseline
    2. Sparsity prevents overfitting
    3. Gate allows model to choose physics vs learning
    4. Jacobian regularization ensures smooth dynamics
    
    Parameters
    ----------
    input_dim : int, optional
        Input feature dimension (auto-detected)
    hidden_dim : int, default=16
        Neural network hidden dimension (keep small!)
    latent_dim : int, default=8
        Latent ODE state dimension
    lr : float, default=0.001
        Learning rate
    epochs : int, default=100
        Training epochs
    lambda_l1 : float, default=0.01
        L1 penalty on neural weights (sparsity)
    lambda_gate : float, default=0.1
        L1 penalty on gate α (prefer physics)
    lambda_jac : float, default=0.001
        Jacobian regularization (smooth dynamics)
    time_steps : int, default=10
        ODE integration steps
    solver : str, default='euler'
        ODE solver ('euler' or 'dopri5' if torchdiffeq available)
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        lr: float = 0.001,
        epochs: int = 100,
        lambda_l1: float = 0.01,
        lambda_gate: float = 0.1,
        lambda_jac: float = 0.001,
        time_steps: int = 10,
        solver: str = 'euler',
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.lambda_l1 = lambda_l1
        self.lambda_gate = lambda_gate
        self.lambda_jac = lambda_jac
        self.time_steps = time_steps
        self.solver = solver
        self.random_state = random_state
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.ode_func = None
        self.decoder = None
        self._fitted = False
        
        # Set random seed
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def _build_model(self, input_dim: int):
        """Build the Hybrid Neural ODE model."""
        # Encoder: Input → Latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.Tanh()
        ).to(self.device)
        
        # Hybrid ODE Function
        self.ode_func = HybridODEFunc(
            dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Decoder: Latent → Probability
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1)
        ).to(self.device)
    
    def _integrate_ode(self, y0: torch.Tensor) -> torch.Tensor:
        """
        Integrate ODE forward in time.
        
        Uses either torchdiffeq (if available) or manual Euler.
        """
        if HAS_TORCHDIFFEQ and self.solver == 'dopri5':
            # Use adaptive solver
            t = torch.linspace(0, 1, 2).to(self.device)
            sol = odeint(self.ode_func, y0, t, method='dopri5')
            return sol[-1]
        else:
            # Manual Euler integration
            y = y0
            dt = 1.0 / self.time_steps
            
            for step in range(self.time_steps):
                t = torch.tensor(step * dt, device=self.device)
                dydt = self.ode_func(t, y)
                y = y + dt * dydt
            
            return y
    
    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_latent: torch.Tensor,
        t_sample: torch.Tensor
    ) -> tuple:
        """
        Compute total loss with all regularization terms.
        
        Loss = MSE + λ₁·L1(weights) + λ₂·L1(α) + λ₃·||∇_y f||²
        
        Returns
        -------
        total_loss, mse_loss, l1_loss, gate_loss, jac_loss
        """
        # MSE loss (prediction error)
        mse_loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
        
        # L1 penalty on neural weights (sparsity)
        l1_loss = self.ode_func.get_l1_penalty()
        
        # L1 penalty on gate (prefer physics)
        gate_loss = self.ode_func.get_gate_penalty()
        
        # Jacobian regularization (smooth dynamics)
        # Sample a few points for efficiency
        n_samples = min(32, y_latent.shape[0])
        indices = torch.randperm(y_latent.shape[0])[:n_samples]
        y_sample = y_latent[indices]
        
        jac_loss = self.ode_func.compute_jacobian_reg(y_sample, t_sample)
        
        # Total loss
        total_loss = (
            mse_loss +
            self.lambda_l1 * l1_loss +
            self.lambda_gate * gate_loss +
            self.lambda_jac * jac_loss
        )
        
        return total_loss, mse_loss, l1_loss, gate_loss, jac_loss
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight=None):
        """
        Train the Hybrid Neural ODE model.
        
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
        
        # Optimizer
        params = list(self.encoder.parameters()) + \
                 list(self.ode_func.parameters()) + \
                 list(self.decoder.parameters())
        optimizer = optim.Adam(params, lr=self.lr)
        
        # Training loop
        self.encoder.train()
        self.ode_func.train()
        self.decoder.train()
        
        print(f"  [HybridODE] Training with physical prior + sparse neural correction")
        print(f"    λ_L1={self.lambda_l1}, λ_gate={self.lambda_gate}, λ_jac={self.lambda_jac}")
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            y0 = self.encoder(X_tensor)
            y_final = self._integrate_ode(y0)
            logits = self.decoder(y_final)
            
            # Compute loss with all regularization
            t_sample = torch.tensor(0.5, device=self.device)
            total_loss, mse_loss, l1_loss, gate_loss, jac_loss = self._compute_loss(
                logits, y_tensor, y0, t_sample
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                alpha_val = self.ode_func.alpha.item()
                print(f"    Epoch {epoch+1}/{self.epochs} | "
                      f"Loss: {total_loss.item():.4f} "
                      f"(MSE: {mse_loss.item():.4f}, "
                      f"L1: {l1_loss.item():.4f}, "
                      f"Gate: {gate_loss.item():.4f}, "
                      f"Jac: {jac_loss.item():.4f}) | "
                      f"α={alpha_val:.4f}")
        
        # Print final gate value
        final_alpha = self.ode_func.alpha.item()
        if final_alpha < 0.1:
            print(f"  [HybridODE] ✓ Physics-dominated (α={final_alpha:.4f} < 0.1)")
        elif final_alpha < 0.5:
            print(f"  [HybridODE] ✓ Balanced physics+neural (α={final_alpha:.4f})")
        else:
            print(f"  [HybridODE] ⚠ Neural-dominated (α={final_alpha:.4f} > 0.5)")
        
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
            y0 = self.encoder(X_tensor)
            y_final = self._integrate_ode(y0)
            logits = self.decoder(y_final)
            
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
            'latent_dim': self.latent_dim,
            'lr': self.lr,
            'epochs': self.epochs,
            'lambda_l1': self.lambda_l1,
            'lambda_gate': self.lambda_gate,
            'lambda_jac': self.lambda_jac,
            'time_steps': self.time_steps,
            'solver': self.solver,
            'random_state': self.random_state,
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_diagnostics(self) -> dict:
        """
        Get model diagnostics.
        
        Returns
        -------
        diagnostics : dict
            Model state and parameters
        """
        if not self._fitted:
            return {}
        
        return {
            'alpha': self.ode_func.alpha.item(),
            'linear_eigenvalues': torch.linalg.eigvals(
                self.ode_func.linear.weight
            ).cpu().numpy().tolist(),
            'neural_sparsity': (
                sum(p.abs().sum().item() for p in self.ode_func.neural_net.parameters()) /
                sum(p.numel() for p in self.ode_func.neural_net.parameters())
            ),
        }


__all__ = ['HybridNeuralODEExpert', 'HybridODEFunc']
