"""
Timm backbone wrapper for Vision Transformers.

This module provides a wrapper around timm models to extract features
and integrate with the dual cross-attention architecture.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional
import math


class TimmBackbone(nn.Module):
    """
    Wrapper for timm Vision Transformer models.
    
    Provides feature extraction from pre-trained ViT/DeiT models
    with customizable output layers for FGVC and Re-ID tasks.
    """
    
    def __init__(self, model_name: str, pretrained: bool = True, 
                 num_classes: int = 1000, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0, img_size: int = 224, **kwargs):
        """
        Initialize timm backbone wrapper.
        
        Args:
            model_name: Name of timm model (e.g., 'vit_base_patch16_224', 'deit_small_patch16_224')
            pretrained: Whether to load pretrained weights
            num_classes: Number of output classes
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate for stochastic depth
            img_size: Input image size
            **kwargs: Additional arguments for timm model
        """
        super().__init__()
        
        # Create timm model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, we'll add our own
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            **kwargs
        )
        
        # Store model configuration
        self.model_name = model_name
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Get model configuration
        self.embed_dim = self.model.embed_dim
        self.num_heads = self.model.blocks[0].attn.num_heads if hasattr(self.model, 'blocks') else 12
        self.patch_size = self.model.patch_embed.patch_size[0] if hasattr(self.model, 'patch_embed') else 16
        
        # Calculate number of patches
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Add custom classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize classifier weights
        if num_classes > 0:
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
            - 'features': Patch embeddings of shape (B, N+1, D) including CLS token
            - 'patch_features': Patch embeddings only of shape (B, N, D)
            - 'cls_token': Class token of shape (B, D)
            - 'logits': Classification logits of shape (B, num_classes)
        """
        batch_size = x.shape[0]
        
        # Forward through backbone (without classification head)
        features = self.model.forward_features(x)  # (B, N+1, D)
        
        # Split CLS token and patch features
        cls_token = features[:, 0]  # (B, D)
        patch_features = features[:, 1:]  # (B, N, D)
        
        # Classification logits
        logits = self.classifier(cls_token)  # (B, num_classes)
        
        outputs = {
            'logits': logits,
            'cls_token': cls_token,
        }
        
        if return_features:
            outputs.update({
                'features': features,  # Full features with CLS token
                'patch_features': patch_features,  # Patch features only
            })
        
        return outputs
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features only (no classification).
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, N+1, D)
        """
        return self.model.forward_features(x)
    
    def get_feature_dim(self) -> int:
        """
        Get the feature dimension of the backbone.
        
        Returns:
            Feature dimension (embedding size)
        """
        return self.embed_dim
    
    def get_num_patches(self) -> int:
        """
        Get the number of patches for the input resolution.
        
        Returns:
            Number of patches (sequence length without CLS token)
        """
        return self.num_patches
    
    def get_num_heads(self) -> int:
        """
        Get the number of attention heads.
        
        Returns:
            Number of attention heads
        """
        return self.num_heads
    
    def get_patch_size(self) -> int:
        """
        Get the patch size.
        
        Returns:
            Patch size in pixels
        """
        return self.patch_size
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights from all transformer blocks.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of attention weight tensors from each layer
        """
        attention_weights = []
        
        def attention_hook(module, input, output):
            # Store attention weights
            if hasattr(module, 'attn_weights'):
                attention_weights.append(module.attn_weights.detach())
        
        # Register hooks on attention modules
        hooks = []
        for name, module in self.model.named_modules():
            if 'attn' in name and hasattr(module, 'num_heads'):
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward_features(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze backbone parameters.
        
        Args:
            freeze: Whether to freeze backbone parameters
        """
        for param in self.model.parameters():
            param.requires_grad = not freeze
    
    def get_layer_lr_decay(self, lr_decay_rate: float = 0.9):
        """
        Get layer-wise learning rate decay for fine-tuning.
        
        Args:
            lr_decay_rate: Decay rate for each layer
            
        Returns:
            Dictionary mapping parameter names to learning rate multipliers
        """
        lr_scales = {}
        num_layers = len(self.model.blocks)
        
        # Patch embedding gets full learning rate
        for name, param in self.model.patch_embed.named_parameters():
            lr_scales[f'model.patch_embed.{name}'] = 1.0
        
        # Position embedding gets full learning rate
        if hasattr(self.model, 'pos_embed'):
            lr_scales['model.pos_embed'] = 1.0
        if hasattr(self.model, 'cls_token'):
            lr_scales['model.cls_token'] = 1.0
        
        # Layer-wise decay for transformer blocks
        for i, block in enumerate(self.model.blocks):
            scale = lr_decay_rate ** (num_layers - i - 1)
            for name, param in block.named_parameters():
                lr_scales[f'model.blocks.{i}.{name}'] = scale
        
        # Classification head gets full learning rate
        for name, param in self.classifier.named_parameters():
            lr_scales[f'classifier.{name}'] = 1.0
        
        return lr_scales