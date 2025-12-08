"""
Time Series Diffusion Denoiser Model.

Main model that combines U-Net and scheduler for:
- Forward diffusion (add noise for training)
- Reverse diffusion (denoise for inference)
- Loss computation

Can be used standalone or wrapped by DiffusionFeatureDenoiser.

Author: QFC System - Diffusion Architecture
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_1d import UNet1D
from .diffusion_scheduler import DiffusionScheduler


class DiffusionDenoiserModel(nn.Module):
    """
    Complete diffusion denoiser for time series.
    
    Wraps U-Net backbone with diffusion scheduler for training and inference.
    
    Args:
        in_channels: Number of input channels
        model_channels: Base channel dimension
        out_channels: Output channels (default: same as in_channels)
        num_timesteps: Total diffusion steps
        beta_schedule: 'linear', 'cosine', or 'quadratic'
        num_res_blocks: Residual blocks per encoder/decoder level
        channel_mult: Channel multipliers per level
        time_dim: Time embedding dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        out_channels: Optional[int] = None,
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_timesteps = num_timesteps
        
        # U-Net backbone
        self.unet = UNet1D(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=self.out_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            time_dim=time_dim,
            dropout=dropout,
        )
        
        # Scheduler (will be moved to same device as model)
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict noise from noisy input.
        
        Args:
            x_t: [B, C, L] noisy input
            t: [B] timestep indices
            
        Returns:
            noise_pred: [B, C, L] predicted noise
        """
        return self.unet(x_t, t)
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x_0: [B, C, L] clean data
            noise: [B, C, L] optional pre-sampled noise
            
        Returns:
            dict with 'loss', 'mse', and optionally other metrics
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # Ensure scheduler is on correct device
        if self.scheduler.device != device:
            self.scheduler.to(device)
        
        # Sample random timesteps
        t = self.scheduler.sample_timesteps(batch_size)
        
        # Add noise to get x_t
        x_t, noise = self.scheduler.add_noise(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.forward(x_t, t)
        
        # MSE loss
        mse_loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': mse_loss,
            'mse': mse_loss.detach(),
        }
    
    @torch.no_grad()
    def denoise(
        self,
        x_noisy: torch.Tensor,
        num_inference_steps: int = 20,
        noise_level: float = 0.3,
        clip_output: bool = True,
    ) -> torch.Tensor:
        """
        Denoise a partially noisy input.
        
        Instead of starting from pure noise (x_T), we add controlled noise
        and run partial reverse diffusion.
        
        Args:
            x_noisy: [B, C, L] input to denoise (may already have some noise)
            num_inference_steps: Number of reverse steps
            noise_level: Amount of noise to add (0 = no noise, 1 = full noise)
            clip_output: Clip output to [-1, 1]
            
        Returns:
            x_denoised: [B, C, L] denoised output
        """
        device = x_noisy.device
        batch_size = x_noisy.shape[0]
        
        # Ensure scheduler on device
        if self.scheduler.device != device:
            self.scheduler.to(device)
        
        # Determine starting timestep based on noise level
        start_t = int(noise_level * self.num_timesteps)
        start_t = max(1, min(start_t, self.num_timesteps - 1))
        
        # Add noise to input
        t_batch = torch.full((batch_size,), start_t, device=device, dtype=torch.long)
        x_t, _ = self.scheduler.add_noise(x_noisy, t_batch)
        
        # Determine timesteps for reverse diffusion
        step_ratio = max(1, start_t // num_inference_steps)
        timesteps = range(start_t, -1, -step_ratio)
        
        # Reverse diffusion
        x = x_t
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.scheduler.p_sample(self, x, t_batch, clip_denoised=clip_output)
        
        return x
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate samples from pure noise.
        
        Args:
            shape: Output shape [B, C, L]
            num_inference_steps: Steps for reverse diffusion
            device: Device to generate on
            
        Returns:
            samples: [B, C, L] generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        if self.scheduler.device != device:
            self.scheduler.to(device)
        
        return self.scheduler.sample(
            self,
            shape,
            num_inference_steps=num_inference_steps,
            clip_denoised=True,
        )
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'num_timesteps': self.num_timesteps,
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'DiffusionDenoiserModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})
        
        model = cls(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            num_timesteps=config.get('num_timesteps', 1000),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


if __name__ == "__main__":
    print("[DiffusionDenoiserModel Test]")
    
    # Create model
    model = DiffusionDenoiserModel(
        in_channels=1,
        model_channels=32,
        num_timesteps=100,
        channel_mult=(1, 2),
    )
    
    # Test training
    x_0 = torch.randn(4, 1, 64)
    losses = model.compute_loss(x_0)
    
    print(f"  Input shape: {x_0.shape}")
    print(f"  Loss: {losses['loss'].item():.4f}")
    
    # Test denoising
    x_denoised = model.denoise(x_0, num_inference_steps=10, noise_level=0.3)
    print(f"  Denoised shape: {x_denoised.shape}")
    
    # Test sampling
    samples = model.sample((4, 1, 64), num_inference_steps=20)
    print(f"  Sample shape: {samples.shape}")
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("[OK] DiffusionDenoiserModel test passed!")
