"""
Evaluation metrics for FGVC and Re-ID tasks.

This module implements accuracy computation for FGVC and
mAP, CMC metrics for Re-ID evaluation.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import warnings


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                    topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """
    Compute top-k accuracy for classification.
    
    Args:
        predictions: Model predictions of shape (B, num_classes)
        targets: Ground truth labels of shape (B,)
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracy values
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        # Get top-k predictions
        _, pred = predictions.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        # Compute accuracy for each k
        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(100.0 / batch_size).item()
            accuracies.append(accuracy)
        
        return accuracies


def compute_reid_metrics(query_features: torch.Tensor, gallery_features: torch.Tensor,
                        query_labels: torch.Tensor, gallery_labels: torch.Tensor,
                        query_cams: torch.Tensor, gallery_cams: torch.Tensor,
                        distance_metric: str = 'euclidean') -> Dict[str, float]:
    """
    Compute Re-ID evaluation metrics (mAP, CMC).
    
    Args:
        query_features: Query feature embeddings of shape (Q, D)
        gallery_features: Gallery feature embeddings of shape (G, D)
        query_labels: Query identity labels of shape (Q,)
        gallery_labels: Gallery identity labels of shape (G,)
        query_cams: Query camera IDs of shape (Q,)
        gallery_cams: Gallery camera IDs of shape (G,)
        distance_metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Dictionary containing mAP and CMC scores
    """
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(query_features, gallery_features, distance_metric)
    
    # Evaluate ranking
    mAP, cmc_scores = evaluate_ranking(
        distance_matrix, query_labels, gallery_labels, 
        query_cams, gallery_cams, max_rank=50
    )
    
    # Return comprehensive metrics
    return {
        'mAP': mAP,
        'CMC-1': cmc_scores[0],
        'CMC-5': cmc_scores[4],
        'CMC-10': cmc_scores[9],
        'CMC-20': cmc_scores[19],
        'CMC-50': cmc_scores[49] if len(cmc_scores) >= 50 else cmc_scores[-1]
    }


def compute_distance_matrix(query_features: torch.Tensor, 
                          gallery_features: torch.Tensor,
                          distance_metric: str = 'euclidean') -> torch.Tensor:
    """
    Compute distance matrix between query and gallery features.
    
    Args:
        query_features: Query feature embeddings of shape (Q, D)
        gallery_features: Gallery feature embeddings of shape (G, D)
        distance_metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Distance matrix of shape (Q, G)
    """
    if distance_metric == 'euclidean':
        # Euclidean distance: ||q - g||^2 = ||q||^2 + ||g||^2 - 2*q^T*g
        q_squared = (query_features ** 2).sum(dim=1, keepdim=True)  # (Q, 1)
        g_squared = (gallery_features ** 2).sum(dim=1, keepdim=True)  # (G, 1)
        
        distance_matrix = q_squared + g_squared.t() - 2 * torch.matmul(query_features, gallery_features.t())
        distance_matrix = torch.clamp(distance_matrix, min=0.0)
        distance_matrix = torch.sqrt(distance_matrix + 1e-12)
        
    elif distance_metric == 'cosine':
        # Cosine distance: 1 - cosine_similarity
        query_norm = torch.nn.functional.normalize(query_features, p=2, dim=1)
        gallery_norm = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
        cosine_sim = torch.matmul(query_norm, gallery_norm.t())
        distance_matrix = 1.0 - cosine_sim
        
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    return distance_matrix


def evaluate_ranking(distance_matrix: torch.Tensor, query_labels: torch.Tensor,
                    gallery_labels: torch.Tensor, query_cams: torch.Tensor,
                    gallery_cams: torch.Tensor, max_rank: int = 50) -> Tuple[float, List[float]]:
    """
    Evaluate ranking performance for Re-ID.
    
    Args:
        distance_matrix: Distance matrix of shape (Q, G)
        query_labels: Query identity labels of shape (Q,)
        gallery_labels: Gallery identity labels of shape (G,)
        query_cams: Query camera IDs of shape (Q,)
        gallery_cams: Gallery camera IDs of shape (G,)
        max_rank: Maximum rank for CMC computation
        
    Returns:
        Tuple of (mAP, CMC_scores)
    """
    num_q, num_g = distance_matrix.shape
    
    if num_g < max_rank:
        max_rank = num_g
        warnings.warn(f"Note: number of gallery samples ({num_g}) is smaller than max_rank ({max_rank}), "
                     f"using max_rank={num_g}")
    
    # Convert to numpy for easier processing
    distance_matrix = distance_matrix.cpu().numpy()
    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()
    query_cams = query_cams.cpu().numpy()
    gallery_cams = gallery_cams.cpu().numpy()
    
    # Sort indices by distance (ascending)
    indices = np.argsort(distance_matrix, axis=1)
    
    # Initialize metrics
    APs = []
    CMC = np.zeros(max_rank)
    
    for q_idx in range(num_q):
        # Get query info
        q_label = query_labels[q_idx]
        q_cam = query_cams[q_idx]
        
        # Get sorted gallery indices for this query
        order = indices[q_idx]
        
        # Remove gallery samples that are invalid
        # (same identity and same camera - these are typically the same image)
        remove = (gallery_labels[order] == q_label) & (gallery_cams[order] == q_cam)
        keep = np.invert(remove)
        
        # Apply filtering
        orig_cmc = (gallery_labels[order] == q_label).astype(np.int32)
        orig_cmc = orig_cmc[keep]
        
        if orig_cmc.size == 0:
            # This query has no valid gallery samples
            continue
        
        # Compute CMC for this query
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        # Accumulate CMC
        if cmc.shape[0] >= max_rank:
            CMC += cmc[:max_rank]
        else:
            CMC[:cmc.shape[0]] += cmc
        
        # Compute Average Precision (AP)
        if orig_cmc.sum() == 0:
            # No positive samples for this query
            continue
        
        # Number of relevant items
        num_rel = orig_cmc.sum()
        
        # Compute precision at each relevant position
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        
        # Average Precision
        AP = tmp_cmc.sum() / num_rel
        APs.append(AP)
    
    if len(APs) == 0:
        raise RuntimeError("No valid query-gallery pairs found!")
    
    # Compute final metrics
    mAP = np.mean(APs)
    CMC = CMC / num_q
    
    return mAP, CMC.tolist()


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training and evaluation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """
    Track multiple metrics during training/evaluation.
    """
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize metric tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metrics = {name: AverageMeter() for name in metric_names}
    
    def update(self, metric_dict: Dict[str, float], n: int = 1):
        """
        Update all metrics.
        
        Args:
            metric_dict: Dictionary of metric values
            n: Number of samples
        """
        for name, value in metric_dict.items():
            if name in self.metrics:
                self.metrics[name].update(value, n)
    
    def get_averages(self) -> Dict[str, float]:
        """
        Get average values for all metrics.
        
        Returns:
            Dictionary of average metric values
        """
        return {name: meter.avg for name, meter in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
    
    def summary(self) -> str:
        """
        Get formatted summary of all metrics.
        
        Returns:
            Formatted string with metric values
        """
        avg_metrics = self.get_averages()
        summary_parts = []
        for name, value in avg_metrics.items():
            if 'accuracy' in name.lower() or 'cmc' in name.lower():
                summary_parts.append(f"{name}: {value:.2f}%")
            else:
                summary_parts.append(f"{name}: {value:.4f}")
        
        return " | ".join(summary_parts)


def compute_class_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                          num_classes: int) -> Dict[int, float]:
    """
    Compute per-class accuracy.
    
    Args:
        predictions: Model predictions of shape (B, num_classes)
        targets: Ground truth labels of shape (B,)
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class_id to accuracy
    """
    with torch.no_grad():
        pred_labels = predictions.argmax(dim=1)
        
        class_accuracies = {}
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_correct = (pred_labels[class_mask] == targets[class_mask]).float().mean()
                class_accuracies[class_id] = class_correct.item() * 100
            else:
                class_accuracies[class_id] = 0.0
        
        return class_accuracies


def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, 
                           num_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Model predictions of shape (B, num_classes)
        targets: Ground truth labels of shape (B,)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    with torch.no_grad():
        pred_labels = predictions.argmax(dim=1)
        
        # Create confusion matrix
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        
        for i in range(len(targets)):
            confusion_matrix[targets[i], pred_labels[i]] += 1
        
        return confusion_matrix