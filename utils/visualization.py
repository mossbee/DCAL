"""
Attention visualization utilities.

This module provides tools for visualizing attention maps
from the dual cross-attention model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2
import os
import seaborn as sns
from matplotlib.patches import Rectangle
import torch.nn.functional as F


def visualize_attention_rollout(image: torch.Tensor, attention_rollout: torch.Tensor,
                              patch_size: int = 16, save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize attention rollout on image.
    
    Args:
        image: Input image tensor of shape (C, H, W)
        attention_rollout: Attention rollout of shape (N,) where N is number of patches
        patch_size: Size of each patch
        save_path: Optional path to save visualization
        
    Returns:
        Visualization as numpy array
    """
    # Convert image to numpy and normalize
    if image.dim() == 4:
        image = image.squeeze(0)
    
    img_np = image.permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Get image dimensions
    H, W = img_np.shape[:2]
    
    # Calculate grid size
    grid_h = H // patch_size
    grid_w = W // patch_size
    
    # Reshape attention to grid
    if len(attention_rollout.shape) == 1:
        attention_grid = attention_rollout.reshape(grid_h, grid_w)
    else:
        attention_grid = attention_rollout
    
    # Create heatmap
    heatmap = create_attention_heatmap(attention_grid, (H, W))
    
    # Overlay on image
    vis = overlay_attention_on_image(img_np, heatmap, alpha=0.6)
    
    if save_path:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Attention Heatmap')
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(vis)
        plt.title('Attention Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return vis


def visualize_glca_attention(image: torch.Tensor, glca_maps: Dict[str, torch.Tensor],
                           patch_size: int = 16, save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize GLCA attention maps.
    
    Args:
        image: Input image tensor
        glca_maps: Dictionary of GLCA attention maps
        patch_size: Size of each patch
        save_path: Optional path to save visualization
        
    Returns:
        Visualization as numpy array
    """
    # Convert image to numpy
    if image.dim() == 4:
        image = image.squeeze(0)
    
    img_np = image.permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    H, W = img_np.shape[:2]
    grid_h = H // patch_size
    grid_w = W // patch_size
    
    if save_path:
        num_blocks = len(glca_maps)
        fig, axes = plt.subplots(2, num_blocks + 1, figsize=(4 * (num_blocks + 1), 8))
        
        if num_blocks == 1:
            axes = axes.reshape(2, -1)
        
        # Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        for idx, (block_name, block_data) in enumerate(glca_maps.items()):
            # Selected regions visualization
            selected_mask = torch.zeros(grid_h * grid_w)
            if 'selected_indices' in block_data:
                selected_indices = block_data['selected_indices']
                if isinstance(selected_indices, torch.Tensor):
                    selected_indices = selected_indices.cpu().numpy()
                # Exclude CLS token (index 0)
                patch_indices = [idx - 1 for idx in selected_indices if idx > 0]
                selected_mask[patch_indices] = 1
            
            selected_grid = selected_mask.reshape(grid_h, grid_w)
            selected_heatmap = create_attention_heatmap(selected_grid, (H, W))
            selected_overlay = overlay_attention_on_image(img_np, selected_heatmap, alpha=0.7)
            
            axes[0, idx + 1].imshow(selected_overlay)
            axes[0, idx + 1].set_title(f'{block_name} - Selected Regions')
            axes[0, idx + 1].axis('off')
            
            # Attention weights visualization
            if 'attention_weights' in block_data:
                attn_weights = block_data['attention_weights']
                if isinstance(attn_weights, torch.Tensor):
                    # Use CLS token attention to patches
                    cls_attention = attn_weights[0, 1:]  # Remove CLS to CLS
                    attn_grid = cls_attention.reshape(grid_h, grid_w)
                    attn_heatmap = create_attention_heatmap(attn_grid, (H, W))
                    attn_overlay = overlay_attention_on_image(img_np, attn_heatmap, alpha=0.6)
                    
                    axes[1, idx + 1].imshow(attn_overlay)
                    axes[1, idx + 1].set_title(f'{block_name} - Attention Weights')
                    axes[1, idx + 1].axis('off')
            else:
                axes[1, idx + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Return combined visualization
    combined_vis = img_np.copy()
    for block_data in glca_maps.values():
        if 'selected_indices' in block_data:
            selected_indices = block_data['selected_indices']
            if isinstance(selected_indices, torch.Tensor):
                selected_indices = selected_indices.cpu().numpy()
            
            # Draw rectangles around selected patches
            for idx in selected_indices:
                if idx > 0:  # Skip CLS token
                    patch_idx = idx - 1
                    row = patch_idx // grid_w
                    col = patch_idx % grid_w
                    
                    y1 = row * patch_size
                    x1 = col * patch_size
                    y2 = min(y1 + patch_size, H)
                    x2 = min(x1 + patch_size, W)
                    
                    cv2.rectangle(combined_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return combined_vis


def create_attention_heatmap(attention_weights: torch.Tensor, 
                           image_size: Tuple[int, int]) -> np.ndarray:
    """
    Create heatmap from attention weights.
    
    Args:
        attention_weights: Attention weights for patches
        image_size: Target image size (H, W)
        
    Returns:
        Heatmap as numpy array
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Normalize attention weights
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    # Resize to image size
    H, W = image_size
    if len(attention_weights.shape) == 2:
        heatmap = cv2.resize(attention_weights, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        # If 1D, reshape to square grid first
        grid_size = int(np.sqrt(len(attention_weights)))
        attention_grid = attention_weights.reshape(grid_size, grid_size)
        heatmap = cv2.resize(attention_grid, (W, H), interpolation=cv2.INTER_CUBIC)
    
    return heatmap


def overlay_attention_on_image(image: np.ndarray, attention_map: np.ndarray,
                             alpha: float = 0.6, colormap: str = 'jet') -> np.ndarray:
    """
    Overlay attention heatmap on image.
    
    Args:
        image: Original image as numpy array
        attention_map: Attention heatmap
        alpha: Blending factor
        colormap: Colormap for heatmap
        
    Returns:
        Overlaid image
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Create colored heatmap
    heatmap_colored = plt.cm.get_cmap(colormap)(attention_map)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Blend images
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid


def plot_attention_comparison(images: List[torch.Tensor], 
                            attention_maps: List[Dict[str, torch.Tensor]],
                            titles: List[str],
                            save_path: Optional[str] = None):
    """
    Plot comparison of attention maps across different methods.
    
    Args:
        images: List of input images
        attention_maps: List of attention map dictionaries
        titles: List of titles for each method
        save_path: Optional path to save plot
    """
    num_images = len(images)
    num_methods = len(attention_maps)
    
    fig, axes = plt.subplots(num_images, num_methods + 1, 
                           figsize=(4 * (num_methods + 1), 4 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for img_idx, image in enumerate(images):
        # Convert image to numpy
        if image.dim() == 4:
            image = image.squeeze(0)
        
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Show original image
        axes[img_idx, 0].imshow(img_np)
        axes[img_idx, 0].set_title('Original')
        axes[img_idx, 0].axis('off')
        
        # Show attention maps for each method
        for method_idx, (attention_dict, title) in enumerate(zip(attention_maps, titles)):
            if 'accumulated_attention' in attention_dict:
                attention = attention_dict['accumulated_attention'][img_idx, 0, 1:]  # CLS to patches
                heatmap = create_attention_heatmap(attention, img_np.shape[:2])
                overlaid = overlay_attention_on_image(img_np, heatmap, alpha=0.6)
                
                axes[img_idx, method_idx + 1].imshow(overlaid)
                axes[img_idx, method_idx + 1].set_title(title)
                axes[img_idx, method_idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_statistics(attention_maps: Dict[str, torch.Tensor], 
                            save_path: Optional[str] = None):
    """
    Plot attention statistics and distributions.
    
    Args:
        attention_maps: Dictionary of attention maps
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract attention data
    if 'accumulated_attention' in attention_maps:
        attention = attention_maps['accumulated_attention']
        
        # CLS token attention distribution
        cls_attention = attention[:, 0, 1:].cpu().numpy()  # (B, N-1)
        
        # Plot attention entropy distribution
        entropies = []
        for i in range(cls_attention.shape[0]):
            attn = cls_attention[i]
            entropy = -(attn * np.log(attn + 1e-8)).sum()
            entropies.append(entropy)
        
        axes[0, 0].hist(entropies, bins=30, alpha=0.7)
        axes[0, 0].set_title('Attention Entropy Distribution')
        axes[0, 0].set_xlabel('Entropy')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot mean attention per patch position
        mean_attention = cls_attention.mean(axis=0)
        grid_size = int(np.sqrt(len(mean_attention)))
        attention_grid = mean_attention.reshape(grid_size, grid_size)
        
        im = axes[0, 1].imshow(attention_grid, cmap='hot')
        axes[0, 1].set_title('Mean Attention per Patch')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot attention variance
        var_attention = cls_attention.var(axis=0)
        var_grid = var_attention.reshape(grid_size, grid_size)
        
        im = axes[1, 0].imshow(var_grid, cmap='hot')
        axes[1, 0].set_title('Attention Variance per Patch')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot attention concentration (max attention per sample)
        max_attention = cls_attention.max(axis=1)
        axes[1, 1].hist(max_attention, bins=30, alpha=0.7)
        axes[1, 1].set_title('Maximum Attention Distribution')
        axes[1, 1].set_xlabel('Max Attention')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class AttentionVisualizer:
    """
    Comprehensive attention visualization class.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize visualizer.
        
        Args:
            model: Dual attention model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def visualize_sample(self, image: torch.Tensor, save_dir: str, 
                        sample_name: str = 'sample'):
        """
        Create comprehensive visualization for a single sample.
        
        Args:
            image: Input image tensor
            save_dir: Directory to save visualizations
            sample_name: Name for this sample
        """
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            # Get attention maps
            attention_maps = self.model.get_attention_maps(image.unsqueeze(0).to(self.device))
            
            # Visualize attention rollout
            if 'accumulated_attention' in attention_maps:
                rollout_vis = visualize_attention_rollout(
                    image, 
                    attention_maps['accumulated_attention'][0, 0, 1:],
                    save_path=os.path.join(save_dir, f'{sample_name}_rollout.png')
                )
            
            # Visualize GLCA attention
            if 'glca_attention_maps' in attention_maps:
                glca_vis = visualize_glca_attention(
                    image,
                    attention_maps['glca_attention_maps'],
                    save_path=os.path.join(save_dir, f'{sample_name}_glca.png')
                )
            
            # Plot attention statistics
            plot_attention_statistics(
                attention_maps,
                save_path=os.path.join(save_dir, f'{sample_name}_stats.png')
            )