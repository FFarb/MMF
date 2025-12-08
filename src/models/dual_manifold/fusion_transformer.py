"""
Dual-Manifold Fusion Transformer.

Performs bidirectional cross-attention between Market and Cognitive manifolds.
Outputs updated tokens for both manifolds and a global context representation.

Architecture:
- Stack of fusion layers, each with:
  - Cog → Market cross-attention
  - Market → Cog cross-attention (optional)
  - Self-attention / MLP on each side

Author: QFC System - Dual-Manifold Architecture
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention layer between two manifolds.
    
    Performs:
        1. query_from_A, attend_to_B → updated_A
        2. Optionally: query_from_B, attend_to_A → updated_B
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.bidirectional = bidirectional
        
        # A queries B (Cog queries Market)
        self.cross_attn_a_to_b = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_a_cross = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_a_ffn = nn.LayerNorm(d_model)
        
        # B queries A (Market queries Cog)
        if bidirectional:
            self.cross_attn_b_to_a = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_b_cross = nn.LayerNorm(d_model)
            self.ffn_b = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            self.norm_b_ffn = nn.LayerNorm(d_model)
    
    def forward(
        self,
        cog_tokens: torch.Tensor,
        market_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention.
        
        Args:
            cog_tokens: [B, N_cog, d_model]
            market_tokens: [B, N_market, d_model]
            
        Returns:
            updated_cog: [B, N_cog, d_model]
            updated_market: [B, N_market, d_model]
        """
        # Cog queries Market
        cog_cross, _ = self.cross_attn_a_to_b(
            query=cog_tokens,
            key=market_tokens,
            value=market_tokens,
        )
        cog_tokens = self.norm_a_cross(cog_tokens + cog_cross)
        cog_tokens = self.norm_a_ffn(cog_tokens + self.ffn_a(cog_tokens))
        
        # Market queries Cog
        if self.bidirectional:
            market_cross, _ = self.cross_attn_b_to_a(
                query=market_tokens,
                key=cog_tokens,
                value=cog_tokens,
            )
            market_tokens = self.norm_b_cross(market_tokens + market_cross)
            market_tokens = self.norm_b_ffn(market_tokens + self.ffn_b(market_tokens))
        
        return cog_tokens, market_tokens


class DualManifoldFusionTransformer(nn.Module):
    """
    Fusion transformer between Market and Cognitive manifolds.
    
    Takes tokens from both manifolds and performs cross-attention fusion.
    Outputs updated tokens and a pooled global context.
    
    Args:
        d_model: Embedding dimension
        num_layers: Number of fusion layers
        num_heads: Attention heads
        dropout: Dropout rate
        bidirectional: Both manifolds query each other
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            for _ in range(num_layers)
        ])
        
        # Global context pooling (from Cog manifold)
        self.context_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        market_tokens: torch.Tensor,
        cog_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse market and cognitive tokens.
        
        Args:
            market_tokens: [B, N_market, d_model]
            cog_tokens: [B, N_cog, d_model]
            
        Returns:
            dict with:
                'updated_market_tokens': [B, N_market, d_model]
                'updated_cog_tokens': [B, N_cog, d_model]
                'global_context': [B, d_model]
        """
        # Apply fusion layers
        for layer in self.fusion_layers:
            cog_tokens, market_tokens = layer(cog_tokens, market_tokens)
        
        # Normalize outputs
        cog_tokens = self.output_norm(cog_tokens)
        market_tokens = self.output_norm(market_tokens)
        
        # Compute global context via attention pooling
        # Attention weights over cog tokens
        attn_weights = self.context_pool(cog_tokens).squeeze(-1)  # [B, N_cog]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        global_context = torch.bmm(
            attn_weights.unsqueeze(1),  # [B, 1, N_cog]
            cog_tokens,  # [B, N_cog, d_model]
        ).squeeze(1)  # [B, d_model]
        
        return {
            'updated_market_tokens': market_tokens,
            'updated_cog_tokens': cog_tokens,
            'global_context': global_context,
        }


if __name__ == "__main__":
    print("[DualManifoldFusionTransformer Test]")
    
    transformer = DualManifoldFusionTransformer(
        d_model=64,
        num_layers=2,
        num_heads=4,
    )
    
    # Test inputs
    market_tokens = torch.randn(4, 128, 64)  # [B, L, d_model]
    cog_tokens = torch.randn(4, 8, 64)  # [B, N_experts+2, d_model]
    
    # Forward
    outputs = transformer(market_tokens, cog_tokens)
    
    print(f"  Market input: {market_tokens.shape}")
    print(f"  Cog input: {cog_tokens.shape}")
    print(f"  Updated market: {outputs['updated_market_tokens'].shape}")
    print(f"  Updated cog: {outputs['updated_cog_tokens'].shape}")
    print(f"  Global context: {outputs['global_context'].shape}")
    print(f"  Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    print("[OK] DualManifoldFusionTransformer test passed!")
