"""
Attention mechanisms for dual cross-attention learning.

This module implements the core attention mechanisms:
- Self-Attention (SA)
- Global-Local Cross-Attention (GLCA)
- Pair-Wise Cross-Attention (PWCA)
"""

from .self_attention import MultiHeadSelfAttention
from .glca import GlobalLocalCrossAttention
from .pwca import PairWiseCrossAttention

__all__ = [
    'MultiHeadSelfAttention',
    'GlobalLocalCrossAttention', 
    'PairWiseCrossAttention'
]