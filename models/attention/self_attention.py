"""
Self-attention mechanism implementation.

This module implements multi-head self-attention as described in the paper,
serving as the baseline attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Implements the standard self-attention mechanism from Vision Transformer,
    where attention is computed as: Attention(Q,K,V) = softmax(QK^T/√d)V
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 attn_drop: float = 0., proj_drop: float = 0.):
        """
        Initialize Multi-Head Self-Attention.
        
        Args:
            dim: Input embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, N, D) where B=batch_size, 
               N=sequence_length, D=embedding_dimension
            return_attention: Whether to return attention weights
               
        Returns:
            Tuple of (output_tensor, attention_weights)
            - output_tensor: Shape (B, N, D)
            - attention_weights: Shape (B, num_heads, N, N) if return_attention else None
        """
        B, N, D = x.shape
        
        # Linear projection to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each has shape (B, num_heads, N, head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        else:
            return x, None


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    
    This is the basic building block used in the self-attention branch.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        """
        Initialize Transformer block.
        
        Args:
            dim: Input embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
        """
        super().__init__()
        
        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=drop)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (B, N, D)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + self.drop_path(attn_out)
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, attn_weights


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).
    """
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: Optional[int] = None, drop: float = 0.):
        """
        Initialize MLP.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension
            out_features: Output feature dimension (default: same as input)
            drop: Dropout rate
        """
        super().__init__()
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    This is the same as the DropConnect impl for EfficientNet, etc networks.
    """
    
    def __init__(self, drop_prob: float = 0.):
        """
        Initialize DropPath.
        
        Args:
            drop_prob: Probability of dropping a path
        """
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic depth."""
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output