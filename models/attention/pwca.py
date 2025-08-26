"""
Pair-Wise Cross-Attention (PWCA) implementation.

This module implements the PWCA mechanism that regularizes attention learning
by treating another image as a distractor during training.

Key insight: PWCA increases training difficulty by contaminating attention scores
with another image, forcing the network to discover more discriminative regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import random


class PairWiseCrossAttention(nn.Module):
    """
    Pair-Wise Cross-Attention mechanism.
    
    Computes attention between query of target image and combined key-value
    from both target and distractor images. Only used during training.
    
    The attention mechanism: f_PWCA(Q1, K_c, V_c) = softmax(Q1 * K_c^T / √d) * V_c
    where K_c = [K1; K2] and V_c = [V1; V2] are concatenated key-value pairs.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        """
        Initialize Pair-Wise Cross-Attention.
        
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
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of pair-wise cross-attention.
        
        Args:
            x1: Target image embeddings of shape (B, N, D)
            x2: Distractor image embeddings of shape (B, N, D)
            
        Returns:
            Output tensor for target image of shape (B, N, D)
        """
        B, N, D = x1.shape
        
        # Compute queries from target image
        q1 = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        
        # Compute keys and values from both images
        k1 = self.k_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        v1 = self.v_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        
        k2 = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        v2 = self.v_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        
        # Combine key-value pairs from both images
        k_combined, v_combined = self._combine_key_value(k1, v1, k2, v2)  # (B, num_heads, 2N, head_dim)
        
        # Compute cross-attention between target queries and combined key-values
        attn = (q1 @ k_combined.transpose(-2, -1)) * self.scale  # (B, num_heads, N, 2N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to combined values
        x = (attn @ v_combined).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def _combine_key_value(self, k1: torch.Tensor, v1: torch.Tensor, 
                          k2: torch.Tensor, v2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine key and value tensors from both images.
        
        Args:
            k1, v1: Key and value tensors from target image of shape (B, num_heads, N, head_dim)
            k2, v2: Key and value tensors from distractor image of shape (B, num_heads, N, head_dim)
            
        Returns:
            Combined key and value tensors of shape (B, num_heads, 2N, head_dim)
        """
        # Concatenate along sequence dimension
        k_combined = torch.cat([k1, k2], dim=2)  # (B, num_heads, 2N, head_dim)
        v_combined = torch.cat([v1, v2], dim=2)  # (B, num_heads, 2N, head_dim)
        
        return k_combined, v_combined
    
    def get_attention_map(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Get attention map for visualization and analysis.
        
        Args:
            x1: Target image embeddings of shape (B, N, D)
            x2: Distractor image embeddings of shape (B, N, D)
            
        Returns:
            Attention map of shape (B, num_heads, N, 2N) showing attention from target to both images
        """
        B, N, D = x1.shape
        
        # Compute queries from target image
        q1 = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute keys from both images
        k1 = self.k_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k2 = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Combine keys
        k_combined = torch.cat([k1, k2], dim=2)  # (B, num_heads, 2N, head_dim)
        
        # Compute attention scores
        attn = (q1 @ k_combined.transpose(-2, -1)) * self.scale  # (B, num_heads, N, 2N)
        attn = F.softmax(attn, dim=-1)
        
        return attn
    
    def analyze_attention_split(self, x1: torch.Tensor, x2: torch.Tensor) -> dict:
        """
        Analyze how attention is split between target and distractor images.
        
        Args:
            x1: Target image embeddings of shape (B, N, D)
            x2: Distractor image embeddings of shape (B, N, D)
            
        Returns:
            Dictionary containing attention analysis
        """
        B, N, D = x1.shape
        
        # Get attention map
        attn = self.get_attention_map(x1, x2)  # (B, num_heads, N, 2N)
        
        # Split attention between target and distractor
        attn_target = attn[:, :, :, :N]  # (B, num_heads, N, N)
        attn_distractor = attn[:, :, :, N:]  # (B, num_heads, N, N)
        
        # Compute attention ratios
        target_ratio = attn_target.sum(dim=-1).mean()  # Average attention to target
        distractor_ratio = attn_distractor.sum(dim=-1).mean()  # Average attention to distractor
        
        # Attention entropy (measure of confusion)
        attn_flat = attn.view(B, self.num_heads, N, -1)
        entropy = -(attn_flat * torch.log(attn_flat + 1e-8)).sum(dim=-1).mean()
        
        return {
            'target_attention_ratio': target_ratio.item(),
            'distractor_attention_ratio': distractor_ratio.item(),
            'attention_entropy': entropy.item(),
            'attention_split_balance': min(target_ratio, distractor_ratio) / max(target_ratio, distractor_ratio)
        }


class PWCABlock(nn.Module):
    """
    Complete PWCA block with normalization and residual connection.
    
    This combines PWCA with layer normalization and residual connections
    similar to a standard transformer block.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm):
        """
        Initialize PWCA block.
        
        Args:
            dim: Input embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
        """
        super().__init__()
        
        # Layer normalization
        self.norm = norm_layer(dim)
        
        # Pair-Wise Cross-Attention
        self.pwca = PairWiseCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # Stochastic depth
        from .self_attention import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PWCA block.
        
        Args:
            x1: Target image embeddings of shape (B, N, D)
            x2: Distractor image embeddings of shape (B, N, D)
            
        Returns:
            Output tensor for target image of shape (B, N, D)
        """
        # PWCA with residual connection
        pwca_out = self.pwca(self.norm(x1), self.norm(x2))
        x1 = x1 + self.drop_path(pwca_out)
        
        return x1


class PairSampler:
    """
    Utility class for sampling image pairs for PWCA training.
    
    Implements different strategies for selecting distractor images.
    """
    
    def __init__(self, strategy: str = 'random', same_class_prob: float = 0.0):
        """
        Initialize pair sampler.
        
        Args:
            strategy: Sampling strategy ('random', 'hard_negative', 'mixed')
            same_class_prob: Probability of sampling from same class (for harder training)
        """
        self.strategy = strategy
        self.same_class_prob = same_class_prob
    
    def sample_pairs(self, batch_data: dict) -> dict:
        """
        Sample image pairs for PWCA training.
        
        Args:
            batch_data: Dictionary containing 'images' and 'labels'
            
        Returns:
            Dictionary with paired images and labels
        """
        images = batch_data['images']  # (B, C, H, W)
        labels = batch_data['labels']  # (B,)
        batch_size = images.size(0)
        
        if self.strategy == 'random':
            # Random shuffling
            indices = torch.randperm(batch_size)
            paired_images = images[indices]
            paired_labels = labels[indices]
            
        elif self.strategy == 'hard_negative':
            # Sample from different classes (harder negatives)
            paired_images = []
            paired_labels = []
            
            for i in range(batch_size):
                current_label = labels[i]
                # Find indices with different labels
                different_class_mask = labels != current_label
                different_indices = torch.where(different_class_mask)[0]
                
                if len(different_indices) > 0:
                    # Sample from different class
                    pair_idx = different_indices[torch.randint(0, len(different_indices), (1,))]
                else:
                    # Fallback to random if no different class available
                    pair_idx = torch.randint(0, batch_size, (1,))
                
                paired_images.append(images[pair_idx])
                paired_labels.append(labels[pair_idx])
            
            paired_images = torch.cat(paired_images, dim=0)
            paired_labels = torch.cat(paired_labels, dim=0)
            
        elif self.strategy == 'mixed':
            # Mix of same and different class pairs
            paired_images = []
            paired_labels = []
            
            for i in range(batch_size):
                if random.random() < self.same_class_prob:
                    # Sample from same class
                    current_label = labels[i]
                    same_class_mask = labels == current_label
                    same_indices = torch.where(same_class_mask)[0]
                    same_indices = same_indices[same_indices != i]  # Exclude self
                    
                    if len(same_indices) > 0:
                        pair_idx = same_indices[torch.randint(0, len(same_indices), (1,))]
                    else:
                        pair_idx = torch.randint(0, batch_size, (1,))
                else:
                    # Sample from different class
                    current_label = labels[i]
                    different_class_mask = labels != current_label
                    different_indices = torch.where(different_class_mask)[0]
                    
                    if len(different_indices) > 0:
                        pair_idx = different_indices[torch.randint(0, len(different_indices), (1,))]
                    else:
                        pair_idx = torch.randint(0, batch_size, (1,))
                
                paired_images.append(images[pair_idx])
                paired_labels.append(labels[pair_idx])
            
            paired_images = torch.cat(paired_images, dim=0)
            paired_labels = torch.cat(paired_labels, dim=0)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
        
        return {
            'images': images,
            'labels': labels,
            'paired_images': paired_images,
            'paired_labels': paired_labels
        }