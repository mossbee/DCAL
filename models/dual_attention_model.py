"""
Main dual cross-attention model implementation.

This module combines the backbone with self-attention, GLCA, and PWCA
mechanisms to form the complete dual cross-attention architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from .backbones.timm_wrapper import TimmBackbone
from .attention.self_attention import TransformerBlock
from .attention.glca import GLCABlock
from .attention.pwca import PWCABlock
from ..utils.attention_rollout import compute_attention_rollout


class DualAttentionModel(nn.Module):
    """
    Complete dual cross-attention model.
    
    Integrates backbone with self-attention, global-local cross-attention,
    and pair-wise cross-attention for fine-grained recognition tasks.
    
    Architecture:
    - Backbone (timm ViT/DeiT) for initial feature extraction
    - Self-Attention branch (L=12 SA blocks) 
    - Global-Local Cross-Attention branch (M=1 GLCA block)
    - Pair-Wise Cross-Attention branch (T=12 PWCA blocks, training only)
    """
    
    def __init__(self, backbone_name: str, num_classes: int, task_type: str = 'fgvc',
                 num_sa_blocks: int = 12, num_glca_blocks: int = 1, num_pwca_blocks: int = 12,
                 embed_dim: int = 768, num_heads: int = 12, top_k_ratio: float = 0.1,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.1, img_size: int = 224,
                 **kwargs):
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
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k_ratio = top_k_ratio
        
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
        self.pwca_blocks = nn.ModuleList([
            PWCABlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=drop_path_rate * (i / max(1, num_pwca_blocks - 1))
            )
            for i in range(num_pwca_blocks)
        ])
        
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
        """Forward pass through PWCA branch."""
        for block in self.pwca_blocks:
            x1 = block(x1, x2)
        return x1
    
    def _compute_attention_rollout(self, attention_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout across all SA layers.
        
        Args:
            attention_weights: List of attention weight tensors from each SA layer
            
        Returns:
            Accumulated attention tensor of shape (B, N, N)
        """
        if not attention_weights:
            # Fallback: return identity matrix
            B = attention_weights[0].size(0) if attention_weights else 1
            N = attention_weights[0].size(-1) if attention_weights else 197  # 224x224 patches + CLS
            device = attention_weights[0].device if attention_weights else torch.device('cpu')
            return torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        return compute_attention_rollout(
            attention_weights,
            head_fusion="mean",
            discard_ratio=0.0,  # No discarding for training stability
            add_residual=True
        )
    
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