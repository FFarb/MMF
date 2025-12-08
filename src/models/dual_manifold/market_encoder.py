"""
Market Token Encoder.

Encodes feature sequences into Market Manifold tokens.
Each token represents market state at a specific time step.

Components can include:
- Base features (returns, frac_diff)
- Advanced stats (Hurst, entropy)
- Tensor-Flex latents
- LaP-SDE latent state
- Denoised features from diffusion

Author: QFC System - Dual-Manifold Architecture
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] input tokens
            
        Returns:
            x: [B, L, d_model] with positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MarketTokenEncoder(nn.Module):
    """
    Encodes market features into tokens for the Market Manifold.
    
    Input: Feature window [B, C_features, L]
    Output: Market tokens [B, L, d_model]
    
    Args:
        d_model: Token embedding dimension
        seq_len: Maximum sequence length
        in_channels: Number of input feature channels
        num_layers: Number of self-attention layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_gqa: Use Grouped-Query Attention
        num_kv_groups: Number of KV groups for GQA
    """
    
    def __init__(
        self,
        d_model: int = 64,
        seq_len: int = 128,
        in_channels: int = 16,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gqa: bool = False,
        num_kv_groups: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Feature projection
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Self-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_gqa:
                # Grouped-Query Attention (placeholder - use standard for now)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
            self.layers.append(layer)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        feature_window: torch.Tensor,
        sde_state: Optional[torch.Tensor] = None,
        tensor_flex: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode features into market tokens.
        
        Args:
            feature_window: [B, C, L] feature tensor
            sde_state: [B, D_sde] optional LaP-SDE latent state
            tensor_flex: [B, D_tf] optional Tensor-Flex latents
            
        Returns:
            market_tokens: [B, L, d_model]
        """
        batch_size = feature_window.shape[0]
        
        # Transpose to [B, L, C]
        x = feature_window.permute(0, 2, 1)  # [B, L, C]
        
        # Project features to d_model
        x = self.input_proj(x)  # [B, L, d_model]
        
        # Add optional state information to all tokens
        if sde_state is not None:
            # Project and add to tokens
            sde_proj = nn.Linear(sde_state.shape[-1], self.d_model).to(x.device)
            sde_emb = sde_proj(sde_state).unsqueeze(1)  # [B, 1, d_model]
            x = x + sde_emb
        
        if tensor_flex is not None:
            tf_proj = nn.Linear(tensor_flex.shape[-1], self.d_model).to(x.device)
            tf_emb = tf_proj(tensor_flex).unsqueeze(1)  # [B, 1, d_model]
            x = x + tf_emb
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Self-attention layers
        for layer in self.layers:
            x = layer(x)
        
        # Normalize
        x = self.norm(x)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.d_model


if __name__ == "__main__":
    print("[MarketTokenEncoder Test]")
    
    encoder = MarketTokenEncoder(
        d_model=64,
        seq_len=128,
        in_channels=16,
        num_layers=1,
        num_heads=4,
    )
    
    # Test input
    features = torch.randn(4, 16, 128)  # [B, C, L]
    
    # Encode
    tokens = encoder(features)
    
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {tokens.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("[OK] MarketTokenEncoder test passed!")
