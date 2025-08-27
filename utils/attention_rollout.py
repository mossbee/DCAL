"""
Attention rollout implementation.

This module implements attention rollout to track how information propagates
from input tokens to higher layers by recursively computing attention across layers.

Based on the official implementation from vit_rollout.py
"""

import torch
import torch.nn as nn
from typing import List, Optional


def compute_attention_rollout(attention_weights: List[torch.Tensor], 
                            head_fusion: str = "mean",
                            discard_ratio: float = 0.0,
                            add_residual: bool = True) -> torch.Tensor:
    """
    Compute attention rollout across multiple transformer layers.
    
    **Attention Rollout Theory:**
    In deep transformers, attention weights in later layers don't directly show 
    which input tokens are important because embeddings become increasingly mixed.
    Attention rollout solves this by recursively multiplying attention matrices
    across layers to track information flow from input to output.
    
    **Mathematical Formulation:**
    Given attention matrices S₁, S₂, ..., Sₗ from L layers:
    1. Normalize each layer: S̄ₗ = αSₗ + (1-α)I (typically α=0.5)
    2. Compute rollout: Ŝ = S̄ₗ ⊗ S̄ₗ₋₁ ⊗ ... ⊗ S̄₁
    3. Result: Ŝ[i,j] = accumulated attention from token j to token i
    
    **Key Insights:**
    - Ŝ[0, :] shows which input tokens the CLS token attends to after all layers
    - Residual connections (identity matrix) account for skip connections
    - Head fusion combines multi-head attention into single matrix per layer
    - Optional discarding removes weak attention paths for visualization
    
    **Usage in GLCA:**
    The first row Ŝ[0, 1:] (CLS attention to patches) identifies the most
    discriminative regions for fine-grained recognition tasks.
    
    Args:
        attention_weights: List of attention matrices from each layer
                          Shape: [(B, num_heads, N, N), ...] for L layers
        head_fusion: Strategy to combine attention heads:
                    - "mean": Average across heads (most common)
                    - "max": Maximum across heads (emphasizes strongest attention)
                    - "min": Minimum across heads (conservative estimate)
        discard_ratio: Fraction of weakest attention paths to zero out (0.0-1.0)
                      Used mainly for visualization, typically 0.0 for training
        add_residual: Whether to add identity matrix (residual connections)
                     Should be True to properly model transformer residuals
        
    Returns:
        Accumulated attention tensor of shape (B, N, N) where:
        - result[b, i, j] = accumulated attention from token j to token i
        - result[b, 0, :] = CLS token attention to all tokens (most important)
    """
    if not attention_weights:
        raise ValueError("attention_weights list cannot be empty")
    
    # Initialize result with identity matrix
    batch_size = attention_weights[0].size(0)
    seq_len = attention_weights[0].size(-1)
    result = torch.eye(seq_len, device=attention_weights[0].device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    with torch.no_grad():
        for attention in attention_weights:
            # Fuse attention heads
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)  # (B, N, N)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]  # (B, N, N)
            elif head_fusion == "min":
                attention_heads_fused = attention.min(dim=1)[0]  # (B, N, N)
            else:
                raise ValueError(f"Unsupported head fusion type: {head_fusion}")
            
            # Optional: discard lowest attention paths (but preserve CLS token)
            if discard_ratio > 0.0:
                attention_heads_fused = _discard_lowest_attention(
                    attention_heads_fused, discard_ratio
                )
            
            # Add residual connection and normalize
            if add_residual:
                I = torch.eye(seq_len, device=attention.device).unsqueeze(0).repeat(batch_size, 1, 1)
                attention_heads_fused = (attention_heads_fused + I) / 2
            
            # Normalize to ensure rows sum to 1
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Accumulate attention: result = attention * result
            result = torch.matmul(attention_heads_fused, result)
    
    return result


def _discard_lowest_attention(attention: torch.Tensor, discard_ratio: float) -> torch.Tensor:
    """
    Discard lowest attention paths while preserving CLS token.
    
    Args:
        attention: Attention weights of shape (B, N, N)
        discard_ratio: Ratio of lowest attention paths to discard
        
    Returns:
        Modified attention weights with lowest paths set to 0
    """
    batch_size, seq_len, _ = attention.shape
    attention_copy = attention.clone()
    
    for b in range(batch_size):
        # Flatten attention matrix for each batch
        flat = attention_copy[b].view(-1)
        
        # Find indices of lowest attention values
        num_discard = int(flat.size(0) * discard_ratio)
        _, indices = flat.topk(num_discard, largest=False)
        
        # Don't discard CLS token connections (first row and column)
        cls_indices = torch.cat([
            torch.arange(seq_len),  # First row
            torch.arange(0, seq_len * seq_len, seq_len)  # First column
        ])
        
        # Remove CLS token indices from discard list
        mask = torch.ones(indices.size(0), dtype=torch.bool)
        for cls_idx in cls_indices:
            mask &= (indices != cls_idx)
        indices = indices[mask]
        
        # Set lowest attention values to 0
        flat[indices] = 0
        attention_copy[b] = flat.view(seq_len, seq_len)
    
    return attention_copy


def normalize_attention_weights(attention_weights: torch.Tensor, 
                              residual_weight: float = 0.5) -> torch.Tensor:
    """
    Normalize attention weights with residual connections.
    
    Implements the normalization: normalized_attn = residual_weight * attn + (1-residual_weight) * I
    
    Args:
        attention_weights: Raw attention weights of shape (B, num_heads, N, N) or (B, N, N)
        residual_weight: Weight for residual connection (default 0.5)
        
    Returns:
        Normalized attention weights of shape (B, N, N)
    """
    if attention_weights.dim() == 4:
        # Average over heads if multi-head attention
        attention_weights = attention_weights.mean(dim=1)
    
    batch_size, seq_len, _ = attention_weights.shape
    device = attention_weights.device
    
    # Create identity matrix
    I = torch.eye(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Apply residual connection
    normalized = residual_weight * attention_weights + (1 - residual_weight) * I
    
    # Normalize rows to sum to 1
    normalized = normalized / normalized.sum(dim=-1, keepdim=True)
    
    return normalized


def get_cls_attention_rollout(rollout_attention: torch.Tensor) -> torch.Tensor:
    """
    Extract CLS token attention from rollout attention matrix.
    
    Args:
        rollout_attention: Rollout attention matrix of shape (B, N, N)
        
    Returns:
        CLS token attention of shape (B, N) representing attention from CLS to all tokens
    """
    # Extract first row (CLS token attention to all tokens)
    cls_attention = rollout_attention[:, 0, :]  # (B, N)
    return cls_attention


def get_patch_attention_rollout(rollout_attention: torch.Tensor) -> torch.Tensor:
    """
    Extract patch attention from rollout attention matrix (excluding CLS token).
    
    Args:
        rollout_attention: Rollout attention matrix of shape (B, N, N)
        
    Returns:
        Patch attention of shape (B, N-1) representing CLS attention to patches only
    """
    # Extract CLS attention to patches (excluding CLS token itself)
    patch_attention = rollout_attention[:, 0, 1:]  # (B, N-1)
    return patch_attention


class AttentionRolloutHook:
    """
    Hook class to collect attention weights from transformer layers.
    
    This class can be used to automatically collect attention weights
    during forward pass for later rollout computation.
    """
    
    def __init__(self):
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self, model: nn.Module, attention_layer_name: str = 'attn'):
        """
        Register forward hooks on attention layers.
        
        Args:
            model: The transformer model
            attention_layer_name: Name pattern to identify attention layers
        """
        self.clear_hooks()
        
        for name, module in model.named_modules():
            if attention_layer_name in name and hasattr(module, 'num_heads'):
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)
    
    def _attention_hook(self, module, input, output):
        """Forward hook to capture attention weights."""
        # This will be called during forward pass
        # The exact implementation depends on the attention module structure
        # For now, we'll store the module for later processing
        if hasattr(module, '_last_attention_weights'):
            self.attention_weights.append(module._last_attention_weights.detach().cpu())
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = []
    
    def get_rollout(self, head_fusion: str = "mean", 
                   discard_ratio: float = 0.0) -> Optional[torch.Tensor]:
        """
        Compute attention rollout from collected weights.
        
        Args:
            head_fusion: How to fuse attention heads
            discard_ratio: Ratio of lowest attention paths to discard
            
        Returns:
            Rollout attention matrix or None if no weights collected
        """
        if not self.attention_weights:
            return None
        
        return compute_attention_rollout(
            self.attention_weights, 
            head_fusion=head_fusion,
            discard_ratio=discard_ratio
        )