"""
1D U-Net Backbone for Diffusion Models.

Implements a U-Net architecture for 1D sequences with:
- Sinusoidal positional encoding for timestep embedding
- Residual blocks with GroupNorm and SiLU activation
- Time embedding injection at each layer
- Downsampling/upsampling with stride 2 convolutions

Input: [B, C, L] (batch, channels, sequence length)
Output: [B, C, L] (predicted noise with same shape)

Author: QFC System - Diffusion Architecture
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps.
    
    Maps timestep t ∈ [0, T] to a d_model dimensional embedding.
    Same encoding used in "Attention Is All You Need" and DDPM papers.
    """
    
    def __init__(self, d_model: int, max_timesteps: int = 10000):
        super().__init__()
        self.d_model = d_model
        
        # Pre-compute frequency bands
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] tensor of timestep indices
            
        Returns:
            embeddings: [B, d_model] timestep embeddings
        """
        timesteps = timesteps.float()
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Pad if d_model is odd
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
            
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU activation, and time embedding injection.
    
    Architecture:
        x → Conv1d → GroupNorm → SiLU → Conv1d → GroupNorm → + time_emb → SiLU → out
        |_________________ skip connection (with projection if needed) _______________|
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
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
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, L] input features
            t_emb: [B, time_dim] time embedding
            
        Returns:
            out: [B, C_out, L] output features
        """
        # First block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding (broadcast over sequence length)
        t = self.time_mlp(t_emb)[:, :, None]  # [B, C_out, 1]
        h = h + t
        
        # Second block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = self.act(h)
        
        # Skip connection
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample by factor of 2 using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2 using transposed convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D U-Net for Diffusion Models.
    
    Architecture:
        - Encoder: progressively downsample with ResidualBlocks
        - Bottleneck: ResidualBlocks at lowest resolution
        - Decoder: progressively upsample with skip connections
        - Time embedding injected at every ResidualBlock
    
    Args:
        in_channels: Number of input channels (e.g., 1 for frac_diff, N for multi-feature)
        model_channels: Base channel count (doubled at each level)
        out_channels: Number of output channels (usually same as in_channels)
        num_res_blocks: Number of residual blocks per level
        channel_mult: Channel multipliers for each level (e.g., [1, 2, 4])
        time_dim: Dimension of time embedding
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
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
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        ch = model_channels
        encoder_channels = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(ch, out_ch, time_dim, num_groups, dropout)
                )
                ch = out_ch
                encoder_channels.append(ch)
            
            if level < self.num_levels - 1:
                self.downsamples.append(Downsample(ch))
                encoder_channels.append(ch)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch, ch, time_dim, num_groups, dropout),
            ResidualBlock(ch, ch, time_dim, num_groups, dropout),
        ])
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            mult = channel_mult[level]
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(
                    ResidualBlock(ch + skip_ch, out_ch, time_dim, num_groups, dropout)
                )
                ch = out_ch
            
            if level > 0:
                self.upsamples.append(Upsample(ch))
        
        # Output projection
        self.output_norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.output_conv = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting noise.
        
        Args:
            x: [B, C, L] noisy input
            timesteps: [B] diffusion timesteps
            
        Returns:
            noise_pred: [B, C, L] predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)  # [B, time_dim]
        
        # Initial projection
        h = self.input_conv(x)  # [B, model_channels, L]
        
        # Encoder
        skips = [h]
        block_idx = 0
        downsample_idx = 0
        
        for level in range(self.num_levels):
            for _ in range(2):  # num_res_blocks
                h = self.encoder_blocks[block_idx](h, t_emb)
                skips.append(h)
                block_idx += 1
            
            if level < self.num_levels - 1:
                h = self.downsamples[downsample_idx](h)
                skips.append(h)
                downsample_idx += 1
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb)
        
        # Decoder
        block_idx = 0
        upsample_idx = 0
        
        for level in reversed(range(self.num_levels)):
            for i in range(3):  # num_res_blocks + 1
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            if level > 0:
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


if __name__ == "__main__":
    # Quick test
    print("[UNet1D Test]")
    
    batch_size = 4
    channels = 1
    seq_len = 128
    
    model = UNet1D(
        in_channels=channels,
        model_channels=64,
        out_channels=channels,
        channel_mult=(1, 2, 4),
    )
    
    x = torch.randn(batch_size, channels, seq_len)
    t = torch.randint(0, 1000, (batch_size,))
    
    out = model(x, t)
    
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print("[OK] UNet1D test passed!")
