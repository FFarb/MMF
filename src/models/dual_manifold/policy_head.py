"""
Dual-Manifold Policy Head.

Takes fused cognitive tokens and global context to produce:
- Expert weights for MoE gating
- Confidence score
- Risk modulation signals

Author: QFC System - Dual-Manifold Architecture
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualManifoldPolicyHead(nn.Module):
    """
    Policy head for generating expert weights from fused representations.
    
    Takes the updated cognitive tokens and global context from the
    fusion transformer and produces:
        - Expert weights (softmax)
        - Confidence score
        - Risk modulation
    
    Args:
        d_model: Embedding dimension
        num_experts: Number of experts (6 with Diffusion)
        temperature: Softmax temperature for weights
        use_confidence: Output confidence score
        use_risk_modulation: Output risk modulation signal
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_experts: int = 6,
        temperature: float = 1.0,
        use_confidence: bool = True,
        use_risk_modulation: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_confidence = use_confidence
        self.use_risk_modulation = use_risk_modulation
        
        # Expert weight head
        # Uses both per-expert tokens and global context
        self.expert_weight_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        
        # Global context to expert logits
        self.context_to_weights = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_experts),
        )
        
        # Confidence head
        if use_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
        
        # Risk modulation head
        if use_risk_modulation:
            self.risk_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 2),  # leverage_scale, position_scale
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        cog_tokens: torch.Tensor,
        global_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute policy outputs from fused representations.
        
        Args:
            cog_tokens: [B, N_cog, d_model] updated cognitive tokens
            global_context: [B, d_model] pooled global representation
            
        Returns:
            dict with:
                'expert_weights': [B, num_experts] softmax weights
                'confidence': [B, 1] confidence score (optional)
                'risk_modulation': [B, 2] risk signals (optional)
        """
        batch_size = cog_tokens.shape[0]
        
        # Method 1: Per-expert attention
        # Assume first num_experts tokens are expert tokens
        expert_tokens = cog_tokens[:, :self.num_experts, :]  # [B, num_experts, d_model]
        
        # Expand global context for concatenation
        global_expanded = global_context.unsqueeze(1).expand(-1, self.num_experts, -1)
        
        # Concatenate expert tokens with global context
        combined = torch.cat([expert_tokens, global_expanded], dim=-1)  # [B, num_experts, d_model*2]
        
        # Get per-expert logits
        expert_logits_per = self.expert_weight_head(combined).squeeze(-1)  # [B, num_experts]
        
        # Method 2: Global context to weights
        expert_logits_global = self.context_to_weights(global_context)  # [B, num_experts]
        
        # Combine both methods
        expert_logits = expert_logits_per + expert_logits_global
        
        # Apply temperature and softmax
        expert_weights = F.softmax(expert_logits / self.temperature, dim=-1)
        
        outputs = {'expert_weights': expert_weights}
        
        # Confidence
        if self.use_confidence:
            confidence = self.confidence_head(global_context)
            outputs['confidence'] = confidence
        
        # Risk modulation
        if self.use_risk_modulation:
            risk_mod = self.risk_head(global_context)
            outputs['risk_modulation'] = risk_mod
        
        return outputs
    
    def set_temperature(self, temperature: float):
        """Update softmax temperature."""
        self.temperature = temperature


class CrossAssetAttentionLayer(nn.Module):
    """
    Cross-asset attention for multi-asset trading.
    
    Allows different assets to attend to each other,
    capturing cross-asset correlations and hedging opportunities.
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
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
    
    def forward(self, asset_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-asset attention.
        
        Args:
            asset_embeddings: [B, N_assets, d_model]
            
        Returns:
            updated_embeddings: [B, N_assets, d_model]
        """
        x = asset_embeddings
        
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


if __name__ == "__main__":
    print("[DualManifoldPolicyHead Test]")
    
    policy_head = DualManifoldPolicyHead(
        d_model=64,
        num_experts=6,
        temperature=1.0,
    )
    
    # Test inputs
    cog_tokens = torch.randn(4, 8, 64)  # [B, N_cog, d_model]
    global_context = torch.randn(4, 64)  # [B, d_model]
    
    # Forward
    outputs = policy_head(cog_tokens, global_context)
    
    print(f"  Expert weights shape: {outputs['expert_weights'].shape}")
    print(f"  Expert weights sum: {outputs['expert_weights'][0].sum().item():.4f}")
    print(f"  Confidence shape: {outputs['confidence'].shape}")
    print(f"  Risk mod shape: {outputs['risk_modulation'].shape}")
    print(f"  Parameters: {sum(p.numel() for p in policy_head.parameters()):,}")
    
    print("[OK] DualManifoldPolicyHead test passed!")
    
    print("\n[CrossAssetAttentionLayer Test]")
    
    cross_asset = CrossAssetAttentionLayer(d_model=64, num_heads=4)
    asset_emb = torch.randn(4, 11, 64)  # [B, N_assets, d_model]
    
    updated = cross_asset(asset_emb)
    print(f"  Input: {asset_emb.shape}")
    print(f"  Output: {updated.shape}")
    
    print("[OK] CrossAssetAttentionLayer test passed!")
