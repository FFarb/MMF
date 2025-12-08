"""
Conditional Diffusion Scenario Model.

Generates future return scenarios conditioned on past features.
Used by DiffusionExpert for probabilistic trading signals.

Shapes:
    - past_window: [B, C_past, L_past] conditioning input
    - future_target: [B, C_target, H] future returns to generate

Author: QFC System - Diffusion Architecture
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_1d import SinusoidalPositionalEncoding, ResidualBlock, Downsample, Upsample
from .diffusion_scheduler import DiffusionScheduler


class ConditionEncoder(nn.Module):
    """
    Encodes past window and regime features into conditioning vector.
    
    Uses 1D CNN followed by global pooling and MLP.
    
    Args:
        in_channels: Number of input channels
        seq_len: Sequence length
        cond_dim: Output conditioning dimension
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        in_channels: int = 16,
        seq_len: int = 96,
        cond_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=7, padding=3),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] past window
            
        Returns:
            cond: [B, cond_dim] conditioning vector
        """
        h = self.conv_layers(x)  # [B, hidden, L']
        h = self.global_pool(h).squeeze(-1)  # [B, hidden]
        cond = self.mlp(h)  # [B, cond_dim]
        return cond


class ConditionalResBlock(nn.Module):
    """
    Residual block with both time and condition injection.
    
    Uses FiLM-style modulation for condition.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        cond_dim: int,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        
        # Condition projection (FiLM-style: scale and shift)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),  # scale + shift
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, L] input
            t_emb: [B, time_dim] time embedding
            cond: [B, cond_dim] condition embedding
            
        Returns:
            out: [B, C_out, L]
        """
        # First block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None]  # [B, C_out, 1]
        h = h + t
        
        # FiLM conditioning
        film = self.cond_mlp(cond)  # [B, C_out * 2]
        scale, shift = film.chunk(2, dim=1)
        scale = scale[:, :, None]  # [B, C_out, 1]
        shift = shift[:, :, None]
        h = h * (1 + scale) + shift
        
        # Second block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = self.act(h)
        
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D U-Net for scenario generation.
    
    Similar to UNet1D but with condition injection at each block.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        out_channels: int = 1,
        cond_dim: int = 128,
        time_dim: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_levels = len(channel_mult)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEncoding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Initial projection
        self.input_conv = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        ch = model_channels
        encoder_channels = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ConditionalResBlock(ch, out_ch, time_dim, cond_dim, num_groups, dropout)
                )
                ch = out_ch
                encoder_channels.append(ch)
            
            if level < self.num_levels - 1:
                self.downsamples.append(Downsample(ch))
                encoder_channels.append(ch)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ConditionalResBlock(ch, ch, time_dim, cond_dim, num_groups, dropout),
            ConditionalResBlock(ch, ch, time_dim, cond_dim, num_groups, dropout),
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            mult = channel_mult[level]
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(
                    ConditionalResBlock(ch + skip_ch, out_ch, time_dim, cond_dim, num_groups, dropout)
                )
                ch = out_ch
            
            if level > 0:
                self.upsamples.append(Upsample(ch))
        
        # Output
        self.output_norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.output_conv = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H] noisy future sequence
            timesteps: [B] diffusion timesteps
            cond: [B, cond_dim] conditioning vector
            
        Returns:
            noise_pred: [B, C, H]
        """
        t_emb = self.time_embed(timesteps)
        h = self.input_conv(x)
        
        # Encoder
        skips = [h]
        block_idx = 0
        downsample_idx = 0
        
        for level in range(self.num_levels):
            for _ in range(2):
                h = self.encoder_blocks[block_idx](h, t_emb, cond)
                skips.append(h)
                block_idx += 1
            
            if level < self.num_levels - 1:
                h = self.downsamples[downsample_idx](h)
                skips.append(h)
                downsample_idx += 1
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb, cond)
        
        # Decoder
        block_idx = 0
        upsample_idx = 0
        
        for level in reversed(range(self.num_levels)):
            for i in range(3):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb, cond)
                block_idx += 1
            
            if level > 0:
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


class DiffusionScenarioModel(nn.Module):
    """
    Conditional diffusion model for generating future return scenarios.
    
    Given a past window of features, generates K possible future trajectories.
    
    Args:
        in_channels_past: Channels in past conditioning window
        in_channels_future: Channels to generate (usually 1 = returns)
        L_past: Length of past conditioning window
        H_future: Forecast horizon
        model_channels: Base model dimension
        cond_dim: Conditioning embedding dimension
        num_timesteps: Diffusion timesteps
        beta_schedule: Beta schedule type
    """
    
    def __init__(
        self,
        in_channels_past: int = 16,
        in_channels_future: int = 1,
        L_past: int = 96,
        H_future: int = 12,
        model_channels: int = 64,
        cond_dim: int = 128,
        time_dim: int = 128,
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels_future = in_channels_future
        self.H_future = H_future
        self.num_timesteps = num_timesteps
        
        # Condition encoder
        self.cond_encoder = ConditionEncoder(
            in_channels=in_channels_past,
            seq_len=L_past,
            cond_dim=cond_dim,
        )
        
        # Conditional U-Net
        self.unet = ConditionalUNet1D(
            in_channels=in_channels_future,
            model_channels=model_channels,
            out_channels=in_channels_future,
            cond_dim=cond_dim,
            time_dim=time_dim,
            channel_mult=(1, 2),  # Shorter sequence, fewer levels
            dropout=dropout,
        )
        
        # Scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        )
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        past_window: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise given noisy future and past conditioning.
        
        Args:
            x_t: [B, C, H] noisy future
            t: [B] timesteps
            past_window: [B, C_past, L_past] conditioning
            
        Returns:
            noise_pred: [B, C, H]
        """
        cond = self.cond_encoder(past_window)
        return self.unet(x_t, t, cond)
    
    def compute_loss(
        self,
        future_window: torch.Tensor,
        past_window: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            future_window: [B, C, H] clean future target
            past_window: [B, C_past, L_past] conditioning
            
        Returns:
            dict with 'loss' and metrics
        """
        device = future_window.device
        batch_size = future_window.shape[0]
        
        if self.scheduler.device != device:
            self.scheduler.to(device)
        
        # Sample timesteps
        t = self.scheduler.sample_timesteps(batch_size)
        
        # Add noise
        x_t, noise = self.scheduler.add_noise(future_window, t)
        
        # Predict noise
        noise_pred = self.forward(x_t, t, past_window)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {'loss': loss, 'mse': loss.detach()}
    
    @torch.no_grad()
    def generate_scenarios(
        self,
        past_window: torch.Tensor,
        num_samples: int = 32,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """
        Generate K future scenarios given past conditioning.
        
        Args:
            past_window: [1, C_past, L_past] or [B, C_past, L_past] conditioning
            num_samples: Number of scenarios to generate per input
            num_inference_steps: Diffusion steps for generation
            
        Returns:
            scenarios: [K, C, H] or [B, K, C, H] generated futures
        """
        device = past_window.device
        
        if self.scheduler.device != device:
            self.scheduler.to(device)
        
        # Handle batching
        if past_window.dim() == 2:
            past_window = past_window.unsqueeze(0)
        
        batch_size = past_window.shape[0]
        
        # Expand conditioning for K samples
        past_expanded = past_window.repeat_interleave(num_samples, dim=0)  # [B*K, C_past, L_past]
        cond = self.cond_encoder(past_expanded)  # [B*K, cond_dim]
        
        # Start from noise
        x = torch.randn(
            batch_size * num_samples,
            self.in_channels_future,
            self.H_future,
            device=device,
        )
        
        # Reverse diffusion
        step_ratio = max(1, self.num_timesteps // num_inference_steps)
        timesteps = range(self.num_timesteps - 1, -1, -step_ratio)
        
        for t in timesteps:
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(x, t_batch, cond)
            
            # Single step
            x = self.scheduler.p_sample(
                lambda xt, tt: self.unet(xt, tt, cond),
                x, t_batch,
                clip_denoised=False,
            )
        
        # Reshape
        if batch_size == 1:
            scenarios = x  # [K, C, H]
        else:
            scenarios = x.view(batch_size, num_samples, self.in_channels_future, self.H_future)
        
        return scenarios


if __name__ == "__main__":
    print("[DiffusionScenarioModel Test]")
    
    # Create model
    model = DiffusionScenarioModel(
        in_channels_past=8,
        in_channels_future=1,
        L_past=64,
        H_future=12,
        model_channels=32,
        cond_dim=64,
        num_timesteps=100,
    )
    
    # Test inputs
    past = torch.randn(4, 8, 64)
    future = torch.randn(4, 1, 12)
    
    # Test training
    losses = model.compute_loss(future, past)
    print(f"  Loss: {losses['loss'].item():.4f}")
    
    # Test generation
    scenarios = model.generate_scenarios(past[:1], num_samples=8, num_inference_steps=10)
    print(f"  Scenarios shape: {scenarios.shape}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    print("[OK] DiffusionScenarioModel test passed!")
