"""
Main dual cross-attention model implementation.

This module combines the backbone with self-attention, GLCA, and PWCA
mechanisms to form the complete dual cross-attention architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from models.backbones.timm_wrapper import TimmBackbone
from models.attention.self_attention import TransformerBlock
from models.attention.glca import GLCABlock
from models.attention.pwca import PWCABlock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.attention_rollout import compute_attention_rollout


class DualAttentionModel(nn.Module):
    """
    Dual Cross-Attention Learning Model for Fine-Grained Visual Recognition.
    
    This implements the complete architecture from "Dual Cross-Attention Learning for 
    Fine-Grained Visual Categorization and Object Re-Identification".
    
    **Architecture Overview:**
    The model extends Vision Transformer with two novel cross-attention mechanisms:
    1. **Global-Local Cross-Attention (GLCA)**: Enhances spatial discrimination
    2. **Pair-Wise Cross-Attention (PWCA)**: Provides attention regularization
    
    **Complete Pipeline:**
    Input Image → Backbone (ViT/DeiT) → Three Parallel Branches:
    ├── Self-Attention Branch (L=12 SA blocks) → SA Classifier
    ├── GLCA Branch (M=1 GLCA block) → GLCA Classifier  
    └── PWCA Branch (T=12 PWCA blocks, training only) → PWCA Classifier
    
    **Key Innovations:**
    
    1. **Attention Rollout Integration:**
       - Computes Ŝᵢ = S̄ᵢ ⊗ S̄ᵢ₋₁ ⊗ ... ⊗ S̄₁ across SA layers
       - Tracks information flow from input patches to final representation
       - Used by GLCA to identify discriminative regions
    
    2. **Global-Local Cross-Attention:**
       - Uses attention rollout to select top-k most important patches
       - Computes cross-attention between selected queries and global key-values
       - Enhances fine-grained spatial discrimination
    
    3. **Pair-Wise Cross-Attention (Training Only):**
       - Introduces controlled distraction during training
       - Target image queries attend to combined target+distractor key-values
       - Regularizes attention learning and prevents overfitting
    
    4. **Uncertainty-Weighted Multi-Task Learning:**
       - Automatically balances losses from three attention branches
       - Uses learnable uncertainty parameters (no manual tuning needed)
       - Adapts loss weights based on task difficulty during training
    
    **Task-Specific Adaptations:**
    
    **Fine-Grained Visual Categorization (FGVC):**
    - Input: 448×448 images (high resolution for fine details)
    - Top-k ratio: R=10% (focus on most discriminative regions)
    - Output: Classification logits for species/categories
    - Evaluation: Top-1 and Top-5 accuracy
    
    **Re-Identification (Re-ID):**
    - Input: 256×256 images (person/vehicle recognition)
    - Top-k ratio: R=30% (broader spatial coverage)
    - Output: Feature embeddings + classification logits
    - Evaluation: mAP and CMC metrics
    - Final features: Concatenated SA + GLCA CLS tokens
    
    **Training Configuration:**
    - Self-Attention: L=12 transformer blocks (shared with PWCA)
    - GLCA: M=1 block (applied after SA rollout computation)
    - PWCA: T=12 blocks (shares weights with SA, training only)
    - Loss weighting: Automatic via uncertainty parameters
    
    **Inference Efficiency:**
    - PWCA branch disabled (no computational overhead)
    - Attention rollout cached for repeated patterns
    - Only SA and GLCA branches active
    
    **Key Benefits:**
    1. **Fine-grained Recognition**: Better discrimination of subtle differences
    2. **Spatial Awareness**: GLCA focuses on discriminative regions
    3. **Regularization**: PWCA prevents overfitting during training
    4. **Automatic Tuning**: Uncertainty weighting eliminates hyperparameter search
    5. **Task Flexibility**: Supports both classification and re-identification
    """
    
    def __init__(self, backbone_name: str, num_classes: int, task_type: str = 'fgvc',
                 num_sa_blocks: int = 12, num_glca_blocks: int = 1, num_pwca_blocks: int = 12,
                 embed_dim: int = 768, num_heads: int = 12, top_k_ratio: float = 0.1,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.1, img_size: int = 224,
                 cache_attention_rollout: bool = True, **kwargs):
        """
        Initialize dual attention model.
        
        Args:
            backbone_name: Name of timm backbone model
            num_classes: Number of output classes
            task_type: Task type ('fgvc' or 'reid')
            num_sa_blocks: Number of self-attention blocks
            num_glca_blocks: Number of GLCA blocks
            num_pwca_blocks: Number of PWCA blocks
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            top_k_ratio: Top-k ratio for GLCA
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
            img_size: Input image size
            cache_attention_rollout: Whether to cache attention rollout for efficiency
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k_ratio = top_k_ratio
        self.cache_attention_rollout = cache_attention_rollout
        
        # Attention rollout cache for efficiency
        self._attention_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Backbone for initial feature extraction
        self.backbone = TimmBackbone(
            model_name=backbone_name,
            pretrained=True,
            num_classes=0,  # Remove head, we'll add our own
            drop_rate=drop_rate,
            img_size=img_size,
            **kwargs
        )
        
        # Update embed_dim from backbone
        self.embed_dim = self.backbone.get_feature_dim()
        
        # Self-Attention branch (shares weights with PWCA)
        self.sa_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=drop_path_rate * (i / max(1, num_sa_blocks - 1))  # Stochastic depth scheduling
            )
            for i in range(num_sa_blocks)
        ])
        
        # Global-Local Cross-Attention branch
        self.glca_blocks = nn.ModuleList([
            GLCABlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                top_k_ratio=top_k_ratio,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=drop_path_rate
            )
            for _ in range(num_glca_blocks)
        ])
        
        # Pair-Wise Cross-Attention branch (training only, shares weights with SA)
        # According to paper: "The PWCA branch shares weights with the SA branch"
        # So we don't create separate PWCA blocks, we reuse SA blocks
        
        # Classification heads for different branches
        self.sa_head = self._create_classifier_head(task_type)
        self.glca_head = self._create_classifier_head(task_type)
        self.pwca_head = self._create_classifier_head(task_type)  # Only used during training
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_classifier_head(self, task_type: str) -> nn.Module:
        """Create task-specific classifier head."""
        if task_type == 'fgvc':
            # Simple linear classifier for FGVC
            return nn.Linear(self.embed_dim, self.num_classes)
        elif task_type == 'reid':
            # Feature extraction for Re-ID (no classification during inference)
            return nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.num_classes)  # For training only
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, x_pair: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of dual attention model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            x_pair: Optional paired input for PWCA training of shape (B, C, H, W)
            
        Returns:
            Dictionary containing:
            - 'sa_logits': Self-attention branch logits
            - 'glca_logits': GLCA branch logits  
            - 'pwca_logits': PWCA branch logits (training only)
            - 'sa_features': SA branch features
            - 'glca_features': GLCA branch features
            - 'combined_features': Combined features for Re-ID
        """
        # Extract initial features from backbone
        backbone_out = self.backbone(x, return_features=True)
        features = backbone_out['features']  # (B, N+1, D) including CLS token
        
        # Self-Attention branch
        sa_features, sa_attention_weights = self._forward_sa_branch(features)
        
        # Compute attention rollout for GLCA
        accumulated_attention = self._compute_attention_rollout(sa_attention_weights)
        
        # Global-Local Cross-Attention branch
        glca_features = self._forward_glca_branch(features, accumulated_attention)
        
        # Pair-Wise Cross-Attention branch (only during training)
        pwca_features = None
        if self.training and x_pair is not None:
            pair_backbone_out = self.backbone(x_pair, return_features=True)
            pair_features = pair_backbone_out['features']
            pwca_features = self._forward_pwca_branch(features, pair_features)
        
        # Extract CLS tokens for classification
        sa_cls = sa_features[:, 0]  # (B, D)
        glca_cls = glca_features[:, 0]  # (B, D)
        
        # Compute logits
        sa_logits = self.sa_head(sa_cls)
        glca_logits = self.glca_head(glca_cls)
        
        outputs = {
            'sa_logits': sa_logits,
            'glca_logits': glca_logits,
            'sa_features': sa_features,
            'glca_features': glca_features,
        }
        
        # PWCA outputs (training only)
        if pwca_features is not None:
            pwca_cls = pwca_features[:, 0]  # (B, D)
            pwca_logits = self.pwca_head(pwca_cls)
            outputs['pwca_logits'] = pwca_logits
            outputs['pwca_features'] = pwca_features
        
        # Combined features for Re-ID
        if self.task_type == 'reid':
            # Concatenate SA and GLCA CLS tokens for final Re-ID features
            combined_features = torch.cat([sa_cls, glca_cls], dim=1)  # (B, 2*D)
            outputs['combined_features'] = combined_features
        
        return outputs
    
    def _forward_sa_branch(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through self-attention branch."""
        attention_weights = []
        
        for block in self.sa_blocks:
            x, attn = block(x, return_attention=True)
            if attn is not None:
                attention_weights.append(attn)
        
        return x, attention_weights
    
    def _forward_glca_branch(self, x: torch.Tensor, accumulated_attention: torch.Tensor) -> torch.Tensor:
        """Forward pass through GLCA branch."""
        for block in self.glca_blocks:
            x = block(x, accumulated_attention)
        return x
    
    def _forward_pwca_branch(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PWCA branch using shared SA weights.
        
        According to paper: "The PWCA branch shares weights with the SA branch"
        This implementation directly uses SA block parameters without weight copying
        for better efficiency.
        
        Args:
            x1: Target image embeddings of shape (B, N, D)
            x2: Distractor image embeddings of shape (B, N, D)
            
        Returns:
            Output tensor for target image of shape (B, N, D)
        """
        for sa_block in self.sa_blocks:
            # Apply layer normalization (shared with SA)
            x1_norm = sa_block.norm1(x1)
            x2_norm = sa_block.norm1(x2)
            
            # Compute PWCA attention using SA block's attention parameters directly
            pwca_output = self._compute_pwca_attention(sa_block.attn, x1_norm, x2_norm)
            
            # Residual connection with stochastic depth
            x1 = x1 + sa_block.drop_path(pwca_output)
            
            # Apply MLP (shared with SA)
            x1 = x1 + sa_block.drop_path(sa_block.mlp(sa_block.norm2(x1)))
        
        return x1
    
    def _compute_pwca_attention(self, attn_module, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute PWCA attention using SA attention module parameters directly.
        
        This avoids weight copying by reusing the SA attention module's QKV projection
        and computing cross-attention between x1 queries and combined x1+x2 key-values.
        
        Args:
            attn_module: The self-attention module from SA block
            x1: Target image embeddings of shape (B, N, D)
            x2: Distractor image embeddings of shape (B, N, D)
            
        Returns:
            PWCA attention output for x1 of shape (B, N, D)
        """
        B, N, D = x1.shape
        
        # Use SA module's QKV projection to get Q, K, V for both images
        # For x1 (target image)
        qkv1 = attn_module.qkv(x1).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1.unbind(0)  # Each: (B, num_heads, N, head_dim)
        
        # For x2 (distractor image)
        qkv2 = attn_module.qkv(x2).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2.unbind(0)  # Each: (B, num_heads, N, head_dim)
        
        # PWCA: Use query from x1, but combined key-value from both x1 and x2
        k_combined = torch.cat([k1, k2], dim=2)  # (B, num_heads, 2N, head_dim)
        v_combined = torch.cat([v1, v2], dim=2)  # (B, num_heads, 2N, head_dim)
        
        # Compute attention scores: q1 @ k_combined^T
        attn = (q1 @ k_combined.transpose(-2, -1)) * attn_module.scale  # (B, num_heads, N, 2N)
        attn = torch.softmax(attn, dim=-1)
        attn = attn_module.attn_drop(attn)
        
        # Apply attention to combined values
        out = (attn @ v_combined).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        
        # Apply output projection (shared with SA)
        out = attn_module.proj(out)
        out = attn_module.proj_drop(out)
        
        return out
    
    def _compute_attention_rollout(self, attention_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout across all SA layers with optional caching.
        
        The attention rollout computation can be expensive, especially during training.
        This method implements caching based on attention weight hashes to avoid
        redundant computations for similar attention patterns.
        
        Args:
            attention_weights: List of attention weight tensors from each SA layer
            
        Returns:
            Accumulated attention tensor of shape (B, N, N)
        """
        if not attention_weights:
            # This should not happen - create identity as fallback but log warning
            print(f"Warning: No attention weights collected, using identity matrix fallback")
            # Use backbone to get proper dimensions
            N = self.backbone.num_patches + 1  # patches + CLS token
            B = 1  # Will be corrected when actually used
            device = next(self.parameters()).device
            return torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # Try to use cache if enabled and we're in eval mode
        if self.cache_attention_rollout and not self.training:
            cache_key = self._compute_attention_cache_key(attention_weights)
            if cache_key in self._attention_cache:
                self._cache_hits += 1
                return self._attention_cache[cache_key]
            else:
                self._cache_misses += 1
        
        # Compute attention rollout
        rollout_result = compute_attention_rollout(
            attention_weights,
            head_fusion="mean",
            discard_ratio=0.0,  # No discarding for training stability
            add_residual=True
        )
        
        # Cache the result if caching is enabled and we're in eval mode
        if self.cache_attention_rollout and not self.training:
            # Limit cache size to prevent memory issues
            if len(self._attention_cache) < 1000:  # Configurable limit
                self._attention_cache[cache_key] = rollout_result.detach()
        
        return rollout_result
    
    def _compute_attention_cache_key(self, attention_weights: List[torch.Tensor]) -> str:
        """
        Compute a hash key for attention weights to enable caching.
        
        This creates a lightweight hash based on attention weight statistics
        rather than the full tensors to balance cache effectiveness with memory usage.
        
        Args:
            attention_weights: List of attention weight tensors
            
        Returns:
            String hash key for caching
        """
        import hashlib
        
        # Create hash based on attention statistics rather than full tensors
        hash_components = []
        
        # Include batch size to prevent cache mismatches
        if attention_weights:
            batch_size = attention_weights[0].size(0)
            hash_components.append(f"batch_{batch_size}")
        
        for i, attn in enumerate(attention_weights):
            # Use attention statistics for hashing (more memory efficient than full tensors)
            attn_mean = attn.mean().item()
            attn_std = attn.std().item()
            attn_max = attn.max().item()
            attn_min = attn.min().item()
            
            # Include layer index and basic statistics
            hash_components.extend([
                f"layer_{i}",
                f"mean_{attn_mean:.6f}",
                f"std_{attn_std:.6f}", 
                f"max_{attn_max:.6f}",
                f"min_{attn_min:.6f}"
            ])
        
        # Create hash from components
        hash_string = "_".join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def clear_attention_cache(self):
        """Clear the attention rollout cache and reset statistics."""
        self._attention_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> dict:
        """Get attention cache statistics for monitoring."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total_requests)
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._attention_cache)
        }
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for inference (Re-ID task).
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor for Re-ID of shape (B, feature_dim)
        """
        with torch.no_grad():
            outputs = self.forward(x, x_pair=None)
            
            if self.task_type == 'reid':
                return outputs['combined_features']
            else:
                # For FGVC, return SA features
                return outputs['sa_features'][:, 0]  # CLS token
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps from different branches for visualization.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing attention maps from different branches
        """
        # Extract backbone features
        backbone_out = self.backbone(x, return_features=True)
        features = backbone_out['features']
        
        # SA branch attention
        sa_features, sa_attention_weights = self._forward_sa_branch(features)
        accumulated_attention = self._compute_attention_rollout(sa_attention_weights)
        
        # GLCA attention maps
        glca_attention_maps = {}
        glca_features_temp = features
        for i, block in enumerate(self.glca_blocks):
            attn_map, selected_indices = block.glca.get_attention_map(glca_features_temp, accumulated_attention)
            glca_attention_maps[f'glca_block_{i}'] = {
                'attention_map': attn_map,
                'selected_indices': selected_indices
            }
            glca_features_temp = block(glca_features_temp, accumulated_attention)
        
        return {
            'sa_attention_weights': sa_attention_weights,
            'accumulated_attention': accumulated_attention,
            'glca_attention_maps': glca_attention_maps,
        }
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters."""
        self.backbone.freeze_backbone(freeze)
    
    def get_parameter_groups(self, lr_backbone: float, lr_attention: float) -> List[Dict]:
        """
        Get parameter groups with different learning rates.
        
        Args:
            lr_backbone: Learning rate for backbone
            lr_attention: Learning rate for attention modules
            
        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = list(self.backbone.parameters())
        attention_params = []
        
        for module in [self.sa_blocks, self.glca_blocks, self.pwca_blocks, 
                      self.sa_head, self.glca_head, self.pwca_head]:
            attention_params.extend(list(module.parameters()))
        
        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': attention_params, 'lr': lr_attention}
        ]