"""
Latent Physics-Informed SDE Expert with Adaptive Dimensionality (LaP-SDE).

This implements a scientifically rigorous approach to stochastic modeling:
- ARD-VAE Encoder: Learns intrinsic market dimensionality (1400 → ~12-20)
- Symbolic Drift Network: Discovers interpretable physics laws
- Diffusion Network: Quantifies market uncertainty
- ELBO Loss: Balances accuracy, simplicity, and physical realism

Architecture:
    Input (1400) → ARD-VAE Encoder → Latent SDE → Decoder → Output
    
    dZ_t = μ_θ(Z_t) dt + σ_φ(Z_t) dW_t
    
Where:
- μ_θ: Symbolic drift (mean reversion, damping, cycles)
- σ_φ: State-dependent diffusion (uncertainty)
- ARD: Automatic Relevance Determination (learns intrinsic dimension)

Loss (ELBO):
    L = L_Rec + β_KL·L_ARD + λ_S·L_Sparse
    
Key Innovation:
- Prediction errors with high σ_pred are penalized less
- This fixes the "Sniper in hiding" problem (Risk Aversion)
- Model learns when to trade (low σ) vs sit out (high σ)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Optional, Dict, Tuple
import warnings
import logging

# Try to import torchsde, fallback to manual Euler-Maruyama if not available
try:
    import torchsde
    HAS_TORCHSDE = True
except ImportError:
    HAS_TORCHSDE = False
    warnings.warn("torchsde not available, using manual Euler-Maruyama integration")

logger = logging.getLogger(__name__)


class SpectralNormLinear(nn.Module):
    """
    Linear layer with Spectral Normalization for Lipschitz continuity.
    
    Enforces smooth, stable transformations in the encoder.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features, bias=bias)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ARDEncoder(nn.Module):
    """
    ARD-VAE Encoder with Automatic Relevance Determination.
    
    Learns to compress 1400 features → L latent dimensions,
    while automatically discovering the intrinsic dimensionality.
    
    ARD Mechanism:
    - Each latent dimension has a learnable prior variance γ_k
    - KL divergence heavily penalizes active dimensions
    - Model "shuts off" useless dimensions to minimize loss
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (e.g., 1400)
    latent_dim : int
        Maximum latent capacity (e.g., 64)
    hidden_dims : list of int
        Hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder network with Spectral Normalization
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                SpectralNormLinear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # ARD: Learnable prior variances (one per latent dimension)
        # Initialized to 1.0 (standard normal prior)
        self.log_prior_var = nn.Parameter(torch.zeros(latent_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution.
        
        Returns
        -------
        z : torch.Tensor
            Sampled latent code
        mu : torch.Tensor
            Mean of q(z|x)
        logvar : torch.Tensor
            Log-variance of q(z|x)
        """
        h = self.encoder(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def get_ard_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute ARD-regularized KL divergence.
        
        KL(q(z|x) || N(0, γ)) = 0.5 * Σ_k [μ²/γ + σ²/γ - log(σ²/γ) - 1]
        
        This heavily penalizes active dimensions, forcing the model
        to use the minimum number of latents necessary.
        """
        prior_var = torch.exp(self.log_prior_var)  # γ_k
        
        # KL divergence per dimension
        kl_per_dim = 0.5 * (
            mu.pow(2) / prior_var +
            torch.exp(logvar) / prior_var -
            logvar + self.log_prior_var - 1
        )
        
        # Sum over latent dimensions, mean over batch
        kl = kl_per_dim.sum(dim=-1).mean()
        
        return kl
    
    def get_active_dimensions(self, threshold: float = 0.01) -> int:
        """
        Count number of "active" latent dimensions.
        
        A dimension is active if its prior variance is significantly
        different from the standard normal (γ_k ≠ 1).
        """
        prior_var = torch.exp(self.log_prior_var)
        active = (torch.abs(prior_var - 1.0) > threshold).sum().item()
        return active


class SymbolicDriftLayer(nn.Module):
    """
    Symbolic Drift Network: Discovers interpretable physics laws.
    
    For each latent dimension z_k, computes candidate terms:
    - Linear: w_1 · z_k (Mean Reversion / Trend)
    - Damping: w_2 · z_k · tanh(z_k) (Stability)
    - Cyclic: w_3 · sin(z_k) (Cycles)
    
    Group sparsity (L1) forces the model to pick one dominant law per dimension.
    
    Parameters
    ----------
    latent_dim : int
        Latent space dimension
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Physics library weights (3 terms per dimension)
        self.w_linear = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.w_damping = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.w_cyclic = nn.Parameter(torch.randn(latent_dim) * 0.1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute drift μ(z) using symbolic physics library.
        
        μ_k(z) = w_1^k · z_k + w_2^k · z_k · tanh(z_k) + w_3^k · sin(z_k)
        """
        # Linear term (mean reversion / trend)
        linear = self.w_linear * z
        
        # Damping term (stability)
        damping = self.w_damping * z * torch.tanh(z)
        
        # Cyclic term (oscillations)
        cyclic = self.w_cyclic * torch.sin(z)
        
        # Combine all terms
        drift = linear + damping + cyclic
        
        return drift
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Compute L1 penalty on physics weights.
        
        Forces the model to use sparse, interpretable equations.
        """
        l1_loss = (
            torch.abs(self.w_linear).sum() +
            torch.abs(self.w_damping).sum() +
            torch.abs(self.w_cyclic).sum()
        )
        return l1_loss
    
    def get_physics_dna(self) -> Dict[int, str]:
        """
        Extract dominant physics law for each dimension.
        
        Returns
        -------
        dna : dict
            Mapping from dimension index to dominant law
        """
        dna = {}
        
        for k in range(self.latent_dim):
            weights = {
                'Linear': abs(self.w_linear[k].item()),
                'Damping': abs(self.w_damping[k].item()),
                'Cyclic': abs(self.w_cyclic[k].item())
            }
            
            # Find dominant term
            dominant = max(weights, key=weights.get)
            
            # Only report if weight is significant
            if weights[dominant] > 0.01:
                dna[k] = dominant
        
        return dna


class DiffusionNetwork(nn.Module):
    """
    Diffusion Network: State-dependent uncertainty σ(z).
    
    Learns how market noise varies with latent state.
    High σ → High uncertainty (don't trade)
    Low σ → Low uncertainty (safe to trade)
    
    Parameters
    ----------
    latent_dim : int
        Latent space dimension
    hidden_dim : int
        Hidden layer dimension
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus()  # Ensure positive output
        )
        
        # Small epsilon for numerical stability
        self.eps = 1e-3
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient σ(z).
        
        Returns strictly positive values.
        """
        sigma = self.net(z) + self.eps
        return sigma


class LatentSDEFunc(nn.Module):
    """
    Latent SDE Function: dZ_t = μ(Z_t) dt + σ(Z_t) dW_t.
    
    Combines symbolic drift and learned diffusion.
    
    Parameters
    ----------
    latent_dim : int
        Latent space dimension
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Drift: Symbolic physics
        self.drift_net = SymbolicDriftLayer(latent_dim)
        
        # Diffusion: Learned uncertainty
        self.diffusion_net = DiffusionNetwork(latent_dim)
        
        # SDE noise type
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Drift term μ(z)."""
        return self.drift_net(z)
    
    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Diffusion term σ(z)."""
        sigma = self.diffusion_net(z)
        
        # For diagonal noise, return (batch, latent_dim)
        return sigma


class LatentPhysicsSDE(nn.Module):
    """
    Complete Latent Physics-Informed SDE Model.
    
    Pipeline:
        Input (1400) → ARD Encoder → Latent SDE → Decoder → Output
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    latent_dim : int
        Maximum latent capacity (will be reduced by ARD)
    hidden_dims : list of int
        Encoder hidden dimensions
    time_steps : int
        SDE integration steps
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: list = None,
        time_steps: int = 10
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        
        # ARD-VAE Encoder
        self.encoder = ARDEncoder(input_dim, latent_dim, hidden_dims)
        
        # Latent SDE
        self.sde_func = LatentSDEFunc(latent_dim)
        
        # Decoder: Latent → Prediction + Uncertainty
        self.decoder_mean = nn.Linear(latent_dim, 1)
        self.decoder_logvar = nn.Linear(latent_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch, input_dim)
        return_uncertainty : bool
            If True, also return prediction uncertainty
        
        Returns
        -------
        y_pred : torch.Tensor
            Predicted logits
        sigma_pred : torch.Tensor (optional)
            Prediction uncertainty
        """
        # Encode to latent
        z0, mu, logvar = self.encoder(x)
        
        # Integrate SDE forward in time
        z_final = self._integrate_sde(z0)
        
        # Decode to prediction
        y_pred = self.decoder_mean(z_final)
        
        if return_uncertainty:
            log_sigma_pred = self.decoder_logvar(z_final)
            sigma_pred = torch.exp(0.5 * log_sigma_pred)
            return y_pred, sigma_pred, mu, logvar
        else:
            return y_pred
    
    def _integrate_sde(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Integrate SDE forward in time using Euler-Maruyama.
        
        dZ_t = μ(Z_t) dt + σ(Z_t) dW_t
        """
        if HAS_TORCHSDE:
            # Use torchsde for efficient integration
            t = torch.linspace(0, 1, 2).to(z0.device)
            
            # torchsde expects (batch, latent_dim) input
            z_traj = torchsde.sdeint(
                self.sde_func,
                z0,
                t,
                method='euler',
                dt=1.0 / self.time_steps
            )
            
            return z_traj[-1]
        else:
            # Manual Euler-Maruyama integration
            z = z0
            dt = 1.0 / self.time_steps
            
            for step in range(self.time_steps):
                t = torch.tensor(step * dt, device=z.device)
                
                # Drift term
                drift = self.sde_func.f(t, z)
                
                # Diffusion term
                diffusion = self.sde_func.g(t, z)
                
                # Brownian increment
                dW = torch.randn_like(z) * np.sqrt(dt)
                
                # Euler-Maruyama step
                z = z + drift * dt + diffusion * dW
            
            return z
    
    def compute_elbo_loss(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        beta_kl: float = 1.0,
        lambda_sparse: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ELBO loss with all components.
        
        Loss = L_Rec + β_KL·L_ARD + λ_S·L_Sparse
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        y_true : torch.Tensor
            True labels
        beta_kl : float
            Weight for KL divergence (ARD penalty)
        lambda_sparse : float
            Weight for sparsity penalty
        
        Returns
        -------
        total_loss : torch.Tensor
            Total ELBO loss
        loss_dict : dict
            Individual loss components
        """
        # Forward pass with uncertainty
        y_pred, sigma_pred, mu, logvar = self.forward(x, return_uncertainty=True)
        
        # 1. Reconstruction Loss (Gaussian NLL)
        # L_Rec = (y - ŷ)² / (2σ²) + log(σ)
        # This is the KEY innovation: errors with high σ are penalized less!
        reconstruction_loss = (
            0.5 * ((y_true - y_pred) ** 2) / (sigma_pred ** 2 + 1e-6) +
            torch.log(sigma_pred + 1e-6)
        ).mean()
        
        # 2. ARD Regularization (KL Divergence)
        # Forces model to use minimum number of latent dimensions
        ard_loss = self.encoder.get_ard_kl(mu, logvar)
        
        # 3. Physics Sparsity
        # L1 norm on drift weights to ensure clean equations
        sparsity_loss = self.sde_func.drift_net.get_sparsity_loss()
        
        # Total ELBO loss
        total_loss = (
            reconstruction_loss +
            beta_kl * ard_loss +
            lambda_sparse * sparsity_loss
        )
        
        # Loss breakdown for telemetry
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': reconstruction_loss.item(),
            'ard_kl': ard_loss.item(),
            'sparsity': sparsity_loss.item(),
            'mean_uncertainty': sigma_pred.mean().item()
        }
        
        return total_loss, loss_dict


class SDEExpert(BaseEstimator, ClassifierMixin):
    """
    Latent Physics-Informed SDE Expert (sklearn-compatible wrapper).
    
    This is the main interface for the MoE ensemble.
    
    Key Features:
    - Separates deterministic trend (drift) from market noise (diffusion)
    - Learns intrinsic market dimensionality via ARD (1400 → ~12-20)
    - Discovers interpretable physics laws (mean reversion, cycles, etc.)
    - Quantifies prediction uncertainty (high σ → don't trade)
    
    Parameters
    ----------
    input_dim : int, optional
        Input feature dimension (auto-detected)
    latent_dim : int, default=64
        Maximum latent capacity (ARD will reduce this)
    hidden_dims : list of int, optional
        Encoder hidden dimensions
    lr : float, default=0.001
        Learning rate
    epochs : int, default=100
        Training epochs
    beta_kl : float, default=1.0
        ARD penalty weight (higher → fewer dimensions)
    lambda_sparse : float, default=0.01
        Physics sparsity weight
    time_steps : int, default=10
        SDE integration steps
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        latent_dim: int = 64,
        hidden_dims: list = None,
        lr: float = 0.001,
        epochs: int = 100,
        beta_kl: float = 1.0,
        lambda_sparse: float = 0.01,
        time_steps: int = 10,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.beta_kl = beta_kl
        self.lambda_sparse = lambda_sparse
        self.time_steps = time_steps
        self.random_state = random_state
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._fitted = False
        
        # Telemetry storage
        self.telemetry = {
            'active_dimensions': [],
            'physics_dna': {},
            'signal_to_noise': [],
            'training_history': []
        }
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight=None):
        """
        Train the Latent Physics-Informed SDE model.
        
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
        self.model = LatentPhysicsSDE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            time_steps=self.time_steps
        ).to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_np).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        
        logger.info(f"[LaP-SDE] Training with ARD-VAE + Physics-Informed SDE")
        logger.info(f"  Input: {self.input_dim} → Latent: {self.latent_dim} (max)")
        logger.info(f"  β_KL={self.beta_kl}, λ_sparse={self.lambda_sparse}")
        
        print(f"\n  [LaP-SDE] Training Latent Physics-Informed SDE")
        print(f"    Input: {self.input_dim} features -> Latent: {self.latent_dim} (max capacity)")
        print(f"    ARD will discover intrinsic dimensionality...")
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Compute ELBO loss
            total_loss, loss_dict = self.model.compute_elbo_loss(
                X_tensor,
                y_tensor,
                beta_kl=self.beta_kl,
                lambda_sparse=self.lambda_sparse
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store telemetry
            self.telemetry['training_history'].append(loss_dict)
            
            # Periodic logging
            if (epoch + 1) % 20 == 0:
                active_dims = self.model.encoder.get_active_dimensions()
                
                print(f"    Epoch {epoch+1}/{self.epochs} | "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"(Rec: {loss_dict['reconstruction']:.4f}, "
                      f"KL: {loss_dict['ard_kl']:.4f}, "
                      f"Sparse: {loss_dict['sparsity']:.4f}) | "
                      f"Active Dims: {active_dims}/{self.latent_dim} | "
                      f"sigma_pred: {loss_dict['mean_uncertainty']:.4f}")
        
        # Final telemetry
        self._compute_final_telemetry()
        
        self._fitted = True
        return self
    
    def _compute_final_telemetry(self):
        """Compute final model telemetry (Latent Prism, Drift DNA, Signal-to-Noise)."""
        
        # 1. Latent Prism: Active Dimensions
        active_dims = self.model.encoder.get_active_dimensions()
        self.telemetry['active_dimensions'] = active_dims
        
        print(f"\n  [LaP-SDE] [OK] Latent Prism: {active_dims}/{self.latent_dim} active dimensions")
        print(f"    Market complexity reduced: {self.input_dim} -> {active_dims} intrinsic units")
        
        # 2. Drift DNA: Physics Laws
        physics_dna = self.model.sde_func.drift_net.get_physics_dna()
        self.telemetry['physics_dna'] = physics_dna
        
        if physics_dna:
            print(f"  [LaP-SDE] [OK] Drift DNA discovered:")
            for dim, law in sorted(physics_dna.items()):
                print(f"    Dim {dim}: {law}")
        else:
            print(f"  [LaP-SDE] [WARNING] No dominant physics laws found (highly stochastic)")
        
        # 3. Signal-to-Noise Ratio
        # Compute ||μ|| / ||σ|| on a sample
        self.model.eval()
        with torch.no_grad():
            # Sample a random latent state
            z_sample = torch.randn(100, self.latent_dim).to(self.device)
            t_sample = torch.tensor(0.5).to(self.device)
            
            drift = self.model.sde_func.f(t_sample, z_sample)
            diffusion = self.model.sde_func.g(t_sample, z_sample)
            
            drift_norm = torch.norm(drift, dim=-1).mean().item()
            diffusion_norm = torch.norm(diffusion, dim=-1).mean().item()
            
            snr = drift_norm / (diffusion_norm + 1e-6)
            self.telemetry['signal_to_noise'] = snr
            
            print(f"  [LaP-SDE] [OK] Signal-to-Noise: {snr:.4f}")
            if snr > 1.0:
                print(f"    Drift-dominated (deterministic trends)")
            elif snr > 0.5:
                print(f"    Balanced drift-diffusion")
            else:
                print(f"    Diffusion-dominated (high uncertainty)")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities with uncertainty quantification.
        
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
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass (no uncertainty needed for prediction)
            y_pred = self.model(X_tensor, return_uncertainty=False)
            
            # Convert to probability
            p_up = torch.sigmoid(y_pred).cpu().numpy().flatten()
        
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
    
    def predict_with_uncertainty(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification.
        
        This is the KEY method for the "Stress-Relax" strategy:
        - High uncertainty → Don't trade (sit out)
        - Low uncertainty → Trade with confidence
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        
        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Class probabilities
        uncertainty : np.ndarray of shape (n_samples,)
            Prediction uncertainty (σ_pred)
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
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with uncertainty
            y_pred, sigma_pred, _, _ = self.model(X_tensor, return_uncertainty=True)
            
            # Convert to probability
            p_up = torch.sigmoid(y_pred).cpu().numpy().flatten()
            uncertainty = sigma_pred.cpu().numpy().flatten()
        
        # Clip to valid range
        p_up = np.clip(p_up, 0.01, 0.99)
        p_down = 1.0 - p_up
        
        proba = np.column_stack([p_down, p_up])
        
        return proba, uncertainty
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'lr': self.lr,
            'epochs': self.epochs,
            'beta_kl': self.beta_kl,
            'lambda_sparse': self.lambda_sparse,
            'time_steps': self.time_steps,
            'random_state': self.random_state,
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_telemetry(self) -> Dict:
        """
        Get comprehensive model telemetry.
        
        Returns
        -------
        telemetry : dict
            - active_dimensions: Number of active latent dimensions
            - physics_dna: Dominant physics law per dimension
            - signal_to_noise: Drift/Diffusion ratio
            - training_history: Loss evolution
        """
        if not self._fitted:
            return {}
        
        return self.telemetry


__all__ = ['SDEExpert', 'LatentPhysicsSDE', 'ARDEncoder', 'SymbolicDriftLayer', 'DiffusionNetwork']
