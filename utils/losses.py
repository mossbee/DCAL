"""
Loss functions for dual cross-attention learning.

This module implements the uncertainty-weighted loss and triplet loss
used for training the dual attention model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-based loss weighting for multi-task learning in dual attention models.
    
    This implements the uncertainty weighting approach from Kendall et al. (2018)
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    adapted for the dual cross-attention architecture.
    
    **Core Innovation:**
    Instead of manually tuning loss weights, this method learns optimal weights
    automatically by modeling the uncertainty (noise) in each task. Tasks with
    higher uncertainty get lower weights, while tasks with lower uncertainty
    get higher weights.
    
    **Mathematical Formulation:**
    For tasks with losses L₁, L₂, L₃ and uncertainty parameters σ₁², σ₂², σ₃²:
    
    L_total = 1/2 * Σᵢ (1/σᵢ² * Lᵢ + log(σᵢ²))
    
    In practice, we parameterize with log-variance: wᵢ = log(σᵢ²)
    So: L_total = 1/2 * Σᵢ (e^(-wᵢ) * Lᵢ + wᵢ)
    
    **Intuition:**
    - e^(-wᵢ) acts as adaptive weight (precision = 1/variance)
    - wᵢ term prevents weights from going to infinity
    - High uncertainty → large wᵢ → small e^(-wᵢ) → low task weight
    - Low uncertainty → small wᵢ → large e^(-wᵢ) → high task weight
    
    **Application to Dual Attention:**
    - L_SA: Self-attention branch loss (baseline performance)
    - L_GLCA: Global-local cross-attention loss (spatial discrimination)
    - L_PWCA: Pair-wise cross-attention loss (regularization, training only)
    
    **Benefits:**
    1. **Automatic Balancing**: No manual hyperparameter tuning needed
    2. **Adaptive Weighting**: Weights adjust during training based on task difficulty
    3. **Principled Approach**: Grounded in Bayesian uncertainty estimation
    4. **Training Stability**: Prevents any single loss from dominating
    
    **Usage Notes:**
    - Initialize log-variance parameters to 0 (equal weighting initially)
    - Monitor weights during training to understand task relationships
    - PWCA loss only included during training (disabled at inference)
    """
    
    def __init__(self, num_tasks: int = 3, init_log_vars: Optional[List[float]] = None):
        """
        Initialize uncertainty weighted loss.
        
        Args:
            num_tasks: Number of tasks/losses to balance (SA, GLCA, PWCA)
            init_log_vars: Initial values for log variance parameters (default: zeros)
        """
        super().__init__()
        self.num_tasks = num_tasks
        
        # Learnable log variance parameters (w1, w2, w3 in the paper)
        if init_log_vars is None:
            init_log_vars = [0.0] * num_tasks
        
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
        
        # Keep track of loss names for consistent ordering
        self.loss_names = ['sa_loss', 'glca_loss', 'pwca_loss'][:num_tasks]
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty weighted total loss.
        
        Args:
            losses: Dictionary of individual losses
                   {'sa_loss': tensor, 'glca_loss': tensor, 'pwca_loss': tensor}
        
        Returns:
            Dictionary containing:
            - 'total_loss': Total weighted loss
            - 'weighted_losses': Individual weighted losses
            - 'weights': Current uncertainty weights
        """
        device = next(iter(losses.values())).device
        total_loss = torch.tensor(0.0, device=device)
        weighted_losses = {}
        weights = {}
        
        # Compute weighted losses
        for i, loss_name in enumerate(self.loss_names):
            if loss_name in losses and losses[loss_name] is not None:
                # Get current log variance
                log_var = self.log_vars[i]
                
                # Compute precision (inverse variance)
                precision = torch.exp(-log_var)
                
                # Weighted loss: precision * loss + log_var
                weighted_loss = precision * losses[loss_name] + log_var
                
                total_loss += weighted_loss
                weighted_losses[loss_name] = weighted_loss
                weights[f'{loss_name}_weight'] = precision
                weights[f'{loss_name}_log_var'] = log_var
        
        # Apply the 1/2 factor from the paper
        total_loss = 0.5 * total_loss
        
        return {
            'total_loss': total_loss,
            'weighted_losses': weighted_losses,
            'weights': weights
        }
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current uncertainty weights for logging.
        
        Returns:
            Dictionary of current weights and log variances
        """
        weights = {}
        with torch.no_grad():
            for i, loss_name in enumerate(self.loss_names):
                log_var = self.log_vars[i].item()
                precision = math.exp(-log_var)
                weights[f'{loss_name}_weight'] = precision
                weights[f'{loss_name}_log_var'] = log_var
        return weights


class TripletLoss(nn.Module):
    """
    Triplet loss for re-identification tasks.
    
    Computes triplet loss to learn discriminative features by pulling
    positive pairs closer and pushing negative pairs apart.
    
    Uses hard negative mining to select the hardest positive and negative
    pairs for more effective training.
    """
    
    def __init__(self, margin: float = 0.3, distance_metric: str = 'euclidean', 
                 hard_mining: bool = True):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ('euclidean' or 'cosine')
            hard_mining: Whether to use hard negative mining
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.hard_mining = hard_mining
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            features: Feature embeddings of shape (B, D)
            labels: Identity labels of shape (B,)
            
        Returns:
            Triplet loss value
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Normalize features for cosine distance
        if self.distance_metric == 'cosine':
            features = F.normalize(features, p=2, dim=1)
        
        # Compute distance matrix
        dist_mat = self._compute_distance_matrix(features)
        
        if self.hard_mining:
            return self._hard_triplet_loss(dist_mat, labels)
        else:
            return self._batch_all_triplet_loss(dist_mat, labels)
    
    def _compute_distance_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix.
        
        Args:
            features: Feature embeddings of shape (B, D)
            
        Returns:
            Distance matrix of shape (B, B)
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
            n = features.size(0)
            
            # Compute squared norms
            feat_squared = (features ** 2).sum(dim=1, keepdim=True)
            
            # Distance matrix
            dist_mat = feat_squared + feat_squared.t() - 2 * torch.matmul(features, features.t())
            
            # Ensure non-negative (numerical stability)
            dist_mat = torch.clamp(dist_mat, min=0.0)
            dist_mat = torch.sqrt(dist_mat + 1e-12)  # Add small epsilon for stability
            
        elif self.distance_metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            # Features are already normalized
            cosine_sim = torch.matmul(features, features.t())
            dist_mat = 1.0 - cosine_sim
            
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return dist_mat
    
    def _hard_triplet_loss(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute hard triplet loss with hard negative mining.
        
        Args:
            dist_mat: Distance matrix of shape (B, B)
            labels: Identity labels of shape (B,)
            
        Returns:
            Hard triplet loss
        """
        N = dist_mat.size(0)
        
        # Create masks for positive and negative pairs
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = ~is_pos
        
        # Set diagonal to False (same sample)
        is_pos = is_pos.fill_diagonal_(False)
        
        # Hard positive mining: for each anchor, find hardest positive
        dist_ap = []
        dist_an = []
        
        for i in range(N):
            # Hardest positive: maximum distance among positives
            pos_mask = is_pos[i]
            if pos_mask.any():
                hard_pos_dist = dist_mat[i][pos_mask].max()
                dist_ap.append(hard_pos_dist)
            else:
                # No positive pairs available
                continue
            
            # Hardest negative: minimum distance among negatives
            neg_mask = is_neg[i]
            if neg_mask.any():
                hard_neg_dist = dist_mat[i][neg_mask].min()
                dist_an.append(hard_neg_dist)
            else:
                # No negative pairs available
                dist_ap.pop()  # Remove the corresponding positive
                continue
        
        if len(dist_ap) == 0:
            return torch.tensor(0.0, device=dist_mat.device, requires_grad=True)
        
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        # Triplet loss: max(0, margin + dist_ap - dist_an)
        triplet_loss = F.relu(self.margin + dist_ap - dist_an).mean()
        
        return triplet_loss
    
    def _batch_all_triplet_loss(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute batch-all triplet loss (all valid triplets).
        
        Args:
            dist_mat: Distance matrix of shape (B, B)
            labels: Identity labels of shape (B,)
            
        Returns:
            Batch-all triplet loss
        """
        N = dist_mat.size(0)
        
        # Create masks
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = ~is_pos
        is_pos = is_pos.fill_diagonal_(False)
        
        # All positive and negative distances
        pos_dist = dist_mat[is_pos]
        neg_dist = dist_mat[is_neg]
        
        if pos_dist.numel() == 0 or neg_dist.numel() == 0:
            return torch.tensor(0.0, device=dist_mat.device, requires_grad=True)
        
        # Compute all valid triplets
        pos_dist = pos_dist.unsqueeze(1)  # (num_pos, 1)
        neg_dist = neg_dist.unsqueeze(0)  # (1, num_neg)
        
        # Triplet loss matrix
        triplet_loss = F.relu(self.margin + pos_dist - neg_dist)
        
        # Only count valid triplets (loss > 0)
        valid_triplets = triplet_loss > 0
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=dist_mat.device, requires_grad=True)
        
        return triplet_loss[valid_triplets].mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for Re-ID training with both cross-entropy and triplet loss.
    
    Combines classification loss (cross-entropy) with metric learning loss (triplet)
    for more effective Re-ID training.
    """
    
    def __init__(self, ce_weight: float = 1.0, triplet_weight: float = 1.0, 
                 triplet_margin: float = 0.3, num_classes: int = 1000):
        """
        Initialize combined loss.
        
        Args:
            ce_weight: Weight for cross-entropy loss
            triplet_weight: Weight for triplet loss
            triplet_margin: Margin for triplet loss
            num_classes: Number of identity classes
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=triplet_margin, hard_mining=True)
    
    def forward(self, logits: torch.Tensor, features: torch.Tensor, 
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: Classification logits of shape (B, num_classes)
            features: Feature embeddings of shape (B, D)
            labels: Identity labels of shape (B,)
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Cross-entropy loss
        ce_loss = self.cross_entropy(logits, labels)
        
        # Triplet loss
        triplet_loss = self.triplet_loss(features, labels)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.triplet_weight * triplet_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'triplet_loss': triplet_loss
        }