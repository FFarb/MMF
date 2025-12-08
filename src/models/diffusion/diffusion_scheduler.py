"""
Diffusion Scheduler for DDPM.

Implements:
- Linear and cosine beta schedules
- Pre-computed alpha, alpha_bar tensors
- Noise sampling utilities
- Forward diffusion q(x_t | x_0)

Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

Author: QFC System - Diffusion Architecture
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DiffusionScheduler:
    """
    Diffusion noise scheduler with pre-computed variance schedules.
    
    Supports:
        - 'linear': Linear beta schedule from beta_start to beta_end
        - 'cosine': Cosine schedule (improved training, less noise at t=0)
        - 'quadratic': Quadratic schedule
    
    Args:
        num_timesteps: Total diffusion steps T
        beta_schedule: Type of schedule ('linear', 'cosine', 'quadratic')
        beta_start: Starting beta value (for linear/quadratic)
        beta_end: Ending beta value (for linear/quadratic)
        device: Torch device
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = 'cpu',
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.device = device
        
        # Compute beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == 'quadratic':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Pre-compute useful quantities
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Square roots for forward diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse diffusion
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clamp to avoid log(0)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".
        
        f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
        alpha_bar(t) = f(t) / f(0)
        beta(t) = 1 - alpha_bar(t) / alpha_bar(t-1)
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean data x_0 to get x_t.
        
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_0: [B, C, L] clean data
            t: [B] timestep indices
            noise: [B, C, L] optional pre-sampled noise
            
        Returns:
            x_t: [B, C, L] noisy data at timestep t
            noise: [B, C, L] the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get coefficients for this timestep
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting: [B] -> [B, 1, 1]
        while sqrt_alpha_bar.dim() < x_0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        # Forward diffusion
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps uniformly."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
    
    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise using:
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        sqrt_recip_alpha_bar = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1_alpha_bar = self.sqrt_recipm1_alphas_cumprod[t]
        
        while sqrt_recip_alpha_bar.dim() < x_t.dim():
            sqrt_recip_alpha_bar = sqrt_recip_alpha_bar.unsqueeze(-1)
            sqrt_recipm1_alpha_bar = sqrt_recipm1_alpha_bar.unsqueeze(-1)
        
        x_0_pred = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * noise_pred
        return x_0_pred
    
    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).
        
        Returns:
            mean: [B, C, L] posterior mean
            variance: [B, C, L] posterior variance
        """
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        variance = self.posterior_variance[t]
        
        while coef1.dim() < x_0.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
            variance = variance.unsqueeze(-1)
        
        mean = coef1 * x_0 + coef2 * x_t
        
        return mean, variance
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion: sample x_{t-1} from p(x_{t-1} | x_t).
        
        Args:
            model: Noise prediction model
            x_t: [B, C, L] current noisy state
            t: [B] current timesteps
            clip_denoised: Whether to clip x_0 prediction to [-1, 1]
            
        Returns:
            x_{t-1}: [B, C, L] sample from posterior
        """
        # Predict noise
        noise_pred = model(x_t, t)
        
        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Get posterior mean and variance
        mean, variance = self.q_posterior(x_0_pred, x_t, t)
        
        # Add noise (except at t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float()
        while nonzero_mask.dim() < x_t.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        sample = mean + nonzero_mask * torch.sqrt(variance) * noise
        return sample
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = True,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion sampling.
        
        Args:
            model: Noise prediction model
            shape: Output shape [B, C, L]
            num_inference_steps: Number of steps (default: all T steps)
            clip_denoised: Clip x_0 predictions
            return_intermediates: Return all intermediate samples
            
        Returns:
            x_0: [B, C, L] generated sample
        """
        device = next(model.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Determine timesteps to use
        if num_inference_steps is None or num_inference_steps >= self.num_timesteps:
            timesteps = range(self.num_timesteps - 1, -1, -1)
        else:
            # DDIM-style subsampling
            step_ratio = self.num_timesteps // num_inference_steps
            timesteps = range(self.num_timesteps - 1, -1, -step_ratio)
        
        intermediates = [x] if return_intermediates else None
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, clip_denoised)
            
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def to(self, device: str):
        """Move all tensors to device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self


if __name__ == "__main__":
    print("[DiffusionScheduler Test]")
    
    # Test scheduler
    scheduler = DiffusionScheduler(num_timesteps=1000, beta_schedule='cosine')
    
    # Test forward diffusion
    x_0 = torch.randn(4, 1, 128)
    t = torch.randint(0, 1000, (4,))
    x_t, noise = scheduler.add_noise(x_0, t)
    
    print(f"  x_0 shape:   {x_0.shape}")
    print(f"  x_t shape:   {x_t.shape}")
    print(f"  noise shape: {noise.shape}")
    print(f"  Beta range:  [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"  Alpha_bar range: [{scheduler.alphas_cumprod.min():.6f}, {scheduler.alphas_cumprod.max():.6f}]")
    print("[OK] DiffusionScheduler test passed!")
