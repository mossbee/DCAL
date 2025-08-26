"""
Global-Local Cross-Attention (GLCA) implementation.

This module implements the GLCA mechanism that enhances interactions between
global images and local high-response regions using attention rollout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.attention_rollout import get_patch_attention_rollout


class GlobalLocalCrossAttention(nn.Module):
    """
    Global-Local Cross-Attention mechanism.
    
    Computes cross-attention between selected local query vectors (high-response regions)
    and global key-value pairs. Uses attention rollout to identify important regions.
    
    Key insight: Instead of treating all queries equally, GLCA selects only the most
    discriminative regions (top-k based on attention rollout) as queries and computes
    cross-attention with the full global key-value set.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0., top_k_ratio: float = 0.1):
        """
        Initialize Global-Local Cross-Attention.
        
        Args:
            dim: Input embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            top_k_ratio: Ratio of top-k regions to select as local queries
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k_ratio = top_k_ratio
        
        # Separate Q, K, V projections for cross-attention
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, accumulated_attention: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of global-local cross-attention.
        
        Args:
            x: Input tensor of shape (B, N, D) where N includes CLS token
            accumulated_attention: Attention rollout weights of shape (B, N, N)
            
        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape
        
        # Extract CLS token attention to patches (for selecting top-k regions)
        cls_attention = get_patch_attention_rollout(accumulated_attention)  # (B, N-1)
        
        # Compute Q, K, V for all tokens
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        
        # Select top-k queries based on attention rollout
        local_q, selected_indices = self._select_top_k_queries(q, cls_attention)  # (B, num_heads, K, head_dim)
        
        # Compute cross-attention between local queries and global key-values
        attn = (local_q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, K, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        local_out = (attn @ v)  # (B, num_heads, K, head_dim)
        
        # Initialize output with original features
        output = x.clone()
        
        # Update selected positions with GLCA output
        local_out = local_out.permute(0, 2, 1, 3).reshape(B, -1, D)  # (B, K, D)
        
        # Scatter local outputs back to their original positions
        for b in range(B):
            output[b, selected_indices[b]] = local_out[b]
        
        # Apply output projection
        output = self.proj(output)
        output = self.proj_drop(output)
        
        return output
    
    def _select_top_k_queries(self, q: torch.Tensor, cls_attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k query vectors based on accumulated attention weights.
        
        Args:
            q: Query tensor of shape (B, num_heads, N, head_dim)
            cls_attention: CLS attention to patches of shape (B, N-1)
            
        Returns:
            Tuple of (selected_queries, selected_indices)
            - selected_queries: Shape (B, num_heads, K, head_dim) where K < N
            - selected_indices: Shape (B, K) indices of selected patches
        """
        B, num_heads, N, head_dim = q.shape
        
        # Calculate number of top-k regions to select
        num_patches = N - 1  # Exclude CLS token
        k = max(1, int(num_patches * self.top_k_ratio))
        
        # Find top-k patches based on CLS attention (excluding CLS token itself)
        _, top_indices = torch.topk(cls_attention, k, dim=-1)  # (B, K)
        
        # Add 1 to indices to account for CLS token at position 0
        patch_indices = top_indices + 1  # (B, K)
        
        # Always include CLS token in selection
        cls_indices = torch.zeros(B, 1, dtype=torch.long, device=q.device)  # (B, 1)
        selected_indices = torch.cat([cls_indices, patch_indices], dim=1)  # (B, K+1)
        
        # Select queries for top-k positions
        # q has shape (B, num_heads, N, head_dim), selected_indices has shape (B, K+1)
        # Use gather to select along the sequence dimension
        selected_indices_expanded = selected_indices.unsqueeze(1).unsqueeze(-1).expand(B, num_heads, k + 1, self.head_dim)  # (B, num_heads, K+1, head_dim)
        selected_q = torch.gather(q, 2, selected_indices_expanded)  # (B, num_heads, K+1, head_dim)
        
        return selected_q, selected_indices
    
    def get_attention_map(self, x: torch.Tensor, accumulated_attention: torch.Tensor) -> torch.Tensor:
        """
        Get attention map for visualization.
        
        Args:
            x: Input tensor of shape (B, N, D)
            accumulated_attention: Attention rollout weights of shape (B, N, N)
            
        Returns:
            Attention map of shape (B, K, N) where K is number of selected queries
        """
        B, N, D = x.shape
        
        # Extract CLS token attention to patches
        cls_attention = get_patch_attention_rollout(accumulated_attention)  # (B, N-1)
        
        # Compute Q, K for attention computation
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Select top-k queries
        local_q, selected_indices = self._select_top_k_queries(q, cls_attention)
        
        # Compute attention scores
        attn = (local_q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, K, N)
        attn = F.softmax(attn, dim=-1)
        
        # Average over heads
        attn = attn.mean(dim=1)  # (B, K, N)
        
        return attn, selected_indices


class GLCABlock(nn.Module):
    """
    Complete GLCA block with normalization and residual connection.
    
    This combines GLCA with layer normalization and residual connections
    similar to a standard transformer block.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, top_k_ratio: float = 0.1,
                 qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        """
        Initialize GLCA block.
        
        Args:
            dim: Input embedding dimension
            num_heads: Number of attention heads
            top_k_ratio: Ratio of top-k regions to select
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
        """
        super().__init__()
        
        # Layer normalization
        self.norm = norm_layer(dim)
        
        # Global-Local Cross-Attention
        self.glca = GlobalLocalCrossAttention(
            dim, num_heads=num_heads, top_k_ratio=top_k_ratio,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        # Stochastic depth
        from models.attention.self_attention import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor, accumulated_attention: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GLCA block.
        
        Args:
            x: Input tensor of shape (B, N, D)
            accumulated_attention: Attention rollout weights of shape (B, N, N)
            
        Returns:
            Output tensor of shape (B, N, D)
        """
        # GLCA with residual connection
        glca_out = self.glca(self.norm(x), accumulated_attention)
        x = x + self.drop_path(glca_out)
        
        return x