"""
Cognitive Token Encoder.

Encodes expert outputs and system state into Cognitive Manifold tokens.
Each token represents a cognitive module (expert, risk state, meta state).

Experts: Trend, Range, Stress, Pattern, Stochastic, Diffusion
Additional: Risk Engine token, Meta/Regime token

Author: QFC System - Dual-Manifold Architecture
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class CognitiveTokenEncoder(nn.Module):
    """
    Encodes expert outputs into Cognitive Manifold tokens.
    
    Each expert gets its own learned embedding based on its outputs.
    Optional Risk and Meta tokens provide system state context.
    
    Args:
        d_model: Token embedding dimension
        num_experts: Number of experts (default 6 with Diffusion)
        use_risk_token: Include Risk Engine state token
        use_meta_token: Include Meta/Regime token
        expert_feature_dim: Input dimension per expert
        num_layers: Self-attention layers within cognitive manifold
        num_heads: Attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_experts: int = 6,
        use_risk_token: bool = True,
        use_meta_token: bool = True,
        expert_feature_dim: int = 8,
        risk_feature_dim: int = 8,
        meta_feature_dim: int = 16,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.use_risk_token = use_risk_token
        self.use_meta_token = use_meta_token
        
        # Calculate total tokens
        self.n_tokens = num_experts
        if use_risk_token:
            self.n_tokens += 1
        if use_meta_token:
            self.n_tokens += 1
        
        # Expert embeddings (one encoder per expert)
        self.expert_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_experts)
        ])
        
        # Expert type embeddings (learned embeddings per expert type)
        self.expert_type_embedding = nn.Embedding(num_experts, d_model)
        
        # Risk token encoder
        if use_risk_token:
            self.risk_encoder = nn.Sequential(
                nn.Linear(risk_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.risk_type_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Meta token encoder
        if use_meta_token:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.meta_type_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Self-attention within cognitive manifold
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        expert_outputs: Dict[str, torch.Tensor],
        risk_state: Optional[torch.Tensor] = None,
        meta_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode expert outputs into cognitive tokens.
        
        Args:
            expert_outputs: Dict with per-expert feature tensors
                Each value should be [B, expert_feature_dim]
                Keys: 'trend', 'range', 'stress', 'pattern', 'stochastic', 'diffusion'
            risk_state: [B, risk_feature_dim] risk engine state
            meta_state: [B, meta_feature_dim] regime/meta state
            
        Returns:
            cog_tokens: [B, N_tokens, d_model]
        """
        # Get batch size from first expert
        first_key = list(expert_outputs.keys())[0]
        batch_size = expert_outputs[first_key].shape[0]
        device = expert_outputs[first_key].device
        
        tokens = []
        
        # Encode each expert
        expert_names = ['trend', 'range', 'stress', 'pattern', 'stochastic', 'diffusion']
        for i, name in enumerate(expert_names[:self.num_experts]):
            if name in expert_outputs:
                expert_feat = expert_outputs[name]
            else:
                # Create placeholder if expert not provided
                expert_feat = torch.zeros(batch_size, 8, device=device)
            
            # Encode expert features
            token = self.expert_encoders[i](expert_feat)  # [B, d_model]
            
            # Add expert type embedding
            type_emb = self.expert_type_embedding(
                torch.tensor([i], device=device)
            ).expand(batch_size, -1)
            token = token + type_emb
            
            tokens.append(token.unsqueeze(1))  # [B, 1, d_model]
        
        # Encode risk state
        if self.use_risk_token:
            if risk_state is not None:
                risk_token = self.risk_encoder(risk_state)
            else:
                risk_token = torch.zeros(batch_size, self.d_model, device=device)
            risk_token = risk_token + self.risk_type_embedding.expand(batch_size, -1, -1).squeeze(1)
            tokens.append(risk_token.unsqueeze(1))
        
        # Encode meta state
        if self.use_meta_token:
            if meta_state is not None:
                meta_token = self.meta_encoder(meta_state)
            else:
                meta_token = torch.zeros(batch_size, self.d_model, device=device)
            meta_token = meta_token + self.meta_type_embedding.expand(batch_size, -1, -1).squeeze(1)
            tokens.append(meta_token.unsqueeze(1))
        
        # Concatenate all tokens
        cog_tokens = torch.cat(tokens, dim=1)  # [B, N_tokens, d_model]
        
        # Self-attention
        for layer in self.layers:
            cog_tokens = layer(cog_tokens)
        
        cog_tokens = self.norm(cog_tokens)
        
        return cog_tokens
    
    def get_num_tokens(self) -> int:
        return self.n_tokens
    
    def get_output_dim(self) -> int:
        return self.d_model


if __name__ == "__main__":
    print("[CognitiveTokenEncoder Test]")
    
    encoder = CognitiveTokenEncoder(
        d_model=64,
        num_experts=6,
        use_risk_token=True,
        use_meta_token=True,
    )
    
    # Create dummy expert outputs
    batch_size = 4
    expert_outputs = {
        'trend': torch.randn(batch_size, 8),
        'range': torch.randn(batch_size, 8),
        'stress': torch.randn(batch_size, 8),
        'pattern': torch.randn(batch_size, 8),
        'stochastic': torch.randn(batch_size, 8),
        'diffusion': torch.randn(batch_size, 8),
    }
    risk_state = torch.randn(batch_size, 8)
    meta_state = torch.randn(batch_size, 16)
    
    # Encode
    tokens = encoder(expert_outputs, risk_state, meta_state)
    
    print(f"  Num tokens: {encoder.get_num_tokens()}")
    print(f"  Output shape: {tokens.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("[OK] CognitiveTokenEncoder test passed!")
