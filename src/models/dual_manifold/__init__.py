"""
Dual-Manifold Fusion Architecture.

This module implements a dual-manifold attention system:
- Market Manifold: Time x Features tokens
- Cognitive Manifold: Expert tokens + Risk/Meta tokens

Cross-attention fusion between manifolds for enhanced decision making.
"""

from .market_encoder import MarketTokenEncoder
from .cog_encoder import CognitiveTokenEncoder
from .fusion_transformer import DualManifoldFusionTransformer
from .policy_head import DualManifoldPolicyHead

__all__ = [
    'MarketTokenEncoder',
    'CognitiveTokenEncoder', 
    'DualManifoldFusionTransformer',
    'DualManifoldPolicyHead',
]
