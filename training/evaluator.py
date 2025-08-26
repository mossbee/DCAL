"""
Evaluation logic for dual cross-attention model.

This module implements evaluation for both FGVC (accuracy) and
Re-ID (mAP, CMC) tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
from ..utils.metrics import (
    compute_accuracy, compute_reid_metrics, compute_class_accuracy, 
    compute_confusion_matrix, MetricTracker
)


class DualAttentionEvaluator:
    """
    Evaluator for dual cross-attention model.
    
    Provides evaluation methods for both FGVC and Re-ID tasks
    with appropriate metrics for each task type.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', task_type: str = 'fgvc'):
        """
        Initialize dual attention evaluator.
        
        Args:
            model: Trained dual attention model
            device: Device for evaluation ('cuda' or 'cpu')
            task_type: Task type ('fgvc' or 'reid')
        """
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_fgvc(self, test_loader: DataLoader, 
                     compute_per_class: bool = True) -> Dict[str, float]:
        """
        Evaluate model on FGVC task.
        
        Args:
            test_loader: Test data loader
            compute_per_class: Whether to compute per-class metrics
            
        Returns:
            Dictionary containing accuracy metrics
        """
        if self.task_type != 'fgvc':
            raise ValueError("This method is only for FGVC tasks")
        
        self.logger.info("Starting FGVC evaluation...")
        
        # Extract predictions
        all_predictions, all_targets = self.compute_predictions(test_loader)
        
        # Compute overall accuracy
        acc1, acc5 = compute_accuracy(all_predictions, all_targets, topk=(1, 5))
        
        results = {
            'accuracy': acc1,
            'top5_accuracy': acc5,
            'num_samples': len(all_targets)
        }
        
        # Compute per-class accuracy
        if compute_per_class:
            num_classes = all_predictions.size(1)
            class_accuracies = compute_class_accuracy(all_predictions, all_targets, num_classes)
            
            # Add class statistics
            results.update({
                'per_class_accuracy': class_accuracies,
                'mean_class_accuracy': np.mean(list(class_accuracies.values())),
                'std_class_accuracy': np.std(list(class_accuracies.values())),
                'min_class_accuracy': min(class_accuracies.values()),
                'max_class_accuracy': max(class_accuracies.values())
            })
        
        # Log results
        self.logger.info(f"FGVC Evaluation Results:")
        self.logger.info(f"  Top-1 Accuracy: {acc1:.2f}%")
        self.logger.info(f"  Top-5 Accuracy: {acc5:.2f}%")
        self.logger.info(f"  Test Samples: {len(all_targets)}")
        
        if compute_per_class:
            self.logger.info(f"  Mean Class Accuracy: {results['mean_class_accuracy']:.2f}%")
            self.logger.info(f"  Std Class Accuracy: {results['std_class_accuracy']:.2f}%")
        
        return results
    
    def evaluate_reid(self, query_loader: DataLoader, 
                     gallery_loader: DataLoader,
                     distance_metric: str = 'euclidean') -> Dict[str, float]:
        """
        Evaluate model on Re-ID task.
        
        Args:
            query_loader: Query set data loader
            gallery_loader: Gallery set data loader
            distance_metric: Distance metric for evaluation
            
        Returns:
            Dictionary containing mAP and CMC metrics
        """
        if self.task_type != 'reid':
            raise ValueError("This method is only for Re-ID tasks")
        
        self.logger.info("Starting Re-ID evaluation...")
        
        # Extract features
        self.logger.info("Extracting query features...")
        query_features, query_labels, query_cams = self.extract_features(query_loader)
        
        self.logger.info("Extracting gallery features...")
        gallery_features, gallery_labels, gallery_cams = self.extract_features(gallery_loader)
        
        self.logger.info(f"Query set: {len(query_features)} samples")
        self.logger.info(f"Gallery set: {len(gallery_features)} samples")
        
        # Compute Re-ID metrics
        results = compute_reid_metrics(
            query_features, gallery_features,
            query_labels, gallery_labels,
            query_cams, gallery_cams,
            distance_metric=distance_metric
        )
        
        # Add dataset statistics
        results.update({
            'num_query': len(query_features),
            'num_gallery': len(gallery_features),
            'num_query_ids': len(torch.unique(query_labels)),
            'num_gallery_ids': len(torch.unique(gallery_labels)),
            'distance_metric': distance_metric
        })
        
        # Log results
        self.logger.info(f"Re-ID Evaluation Results:")
        self.logger.info(f"  mAP: {results['mAP']:.1%}")
        self.logger.info(f"  CMC-1: {results['CMC-1']:.1%}")
        self.logger.info(f"  CMC-5: {results['CMC-5']:.1%}")
        self.logger.info(f"  CMC-10: {results['CMC-10']:.1%}")
        self.logger.info(f"  Query IDs: {results['num_query_ids']}")
        self.logger.info(f"  Gallery IDs: {results['num_gallery_ids']}")
        
        return results
    
    def extract_features(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from data loader.
        
        Args:
            data_loader: Data loader to extract features from
            
        Returns:
            Tuple of (features, labels, camera_ids)
        """
        all_features = []
        all_labels = []
        all_cams = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get camera IDs (for Re-ID datasets)
                if 'camera_id' in batch:
                    cams = batch['camera_id'].to(self.device)
                else:
                    cams = torch.zeros_like(labels)  # Dummy camera IDs
                
                # Extract features
                if self.task_type == 'reid':
                    features = self.model.extract_features(images)
                else:
                    # For FGVC, use SA branch CLS token
                    outputs = self.model(images, x_pair=None)
                    features = outputs['sa_features'][:, 0]  # CLS token
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
                all_cams.append(cams.cpu())
        
        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_cams = torch.cat(all_cams, dim=0)
        
        return all_features, all_labels, all_cams
    
    def compute_predictions(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predictions for FGVC evaluation.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Tuple of (predictions, targets)
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing predictions"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, x_pair=None)
                
                # Use SA branch predictions (can also combine with GLCA)
                predictions = outputs['sa_logits']
                
                # Optionally combine SA and GLCA predictions
                if 'glca_logits' in outputs:
                    glca_predictions = outputs['glca_logits']
                    predictions = (predictions + glca_predictions) / 2
                
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return all_predictions, all_targets
    
    def analyze_attention_patterns(self, data_loader: DataLoader, 
                                 num_samples: int = 100,
                                 save_visualizations: bool = False,
                                 save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze learned attention patterns.
        
        Args:
            data_loader: Data loader for analysis
            num_samples: Number of samples to analyze
            save_visualizations: Whether to save attention visualizations
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary containing attention analysis results
        """
        self.logger.info(f"Analyzing attention patterns on {num_samples} samples...")
        
        attention_stats = {
            'sa_attention_entropy': [],
            'glca_selected_regions': [],
            'attention_diversity': []
        }
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if sample_count >= num_samples:
                    break
                
                images = batch['image'].to(self.device)
                batch_size = images.size(0)
                
                # Get attention maps
                attention_maps = self.model.get_attention_maps(images)
                
                # Analyze SA attention patterns
                sa_attention = attention_maps['accumulated_attention']  # (B, N, N)
                
                for i in range(min(batch_size, num_samples - sample_count)):
                    # Compute attention entropy (measure of attention spread)
                    cls_attention = sa_attention[i, 0, 1:]  # CLS to patches
                    entropy = -(cls_attention * torch.log(cls_attention + 1e-8)).sum()
                    attention_stats['sa_attention_entropy'].append(entropy.item())
                    
                    # Analyze GLCA selected regions
                    if 'glca_attention_maps' in attention_maps:
                        glca_maps = attention_maps['glca_attention_maps']
                        for block_name, block_data in glca_maps.items():
                            selected_indices = block_data['selected_indices'][i]
                            num_selected = len(selected_indices) - 1  # Exclude CLS
                            attention_stats['glca_selected_regions'].append(num_selected)
                    
                    sample_count += 1
        
        # Compute statistics
        results = {
            'num_samples_analyzed': sample_count,
            'mean_sa_entropy': np.mean(attention_stats['sa_attention_entropy']),
            'std_sa_entropy': np.std(attention_stats['sa_attention_entropy']),
            'mean_glca_regions': np.mean(attention_stats['glca_selected_regions']) if attention_stats['glca_selected_regions'] else 0,
            'attention_entropy_distribution': attention_stats['sa_attention_entropy']
        }
        
        self.logger.info(f"Attention Analysis Results:")
        self.logger.info(f"  Mean SA Entropy: {results['mean_sa_entropy']:.3f}")
        self.logger.info(f"  Std SA Entropy: {results['std_sa_entropy']:.3f}")
        self.logger.info(f"  Mean GLCA Regions: {results['mean_glca_regions']:.1f}")
        
        return results
    
    def compare_branches(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Compare performance of different attention branches.
        
        Args:
            data_loader: Data loader for comparison
            
        Returns:
            Dictionary containing branch comparison results
        """
        self.logger.info("Comparing attention branch performance...")
        
        sa_correct = 0
        glca_correct = 0
        combined_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Comparing branches"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, x_pair=None)
                
                # Get predictions from each branch
                sa_pred = outputs['sa_logits'].argmax(dim=1)
                glca_pred = outputs['glca_logits'].argmax(dim=1)
                
                # Combined prediction (average)
                combined_logits = (outputs['sa_logits'] + outputs['glca_logits']) / 2
                combined_pred = combined_logits.argmax(dim=1)
                
                # Count correct predictions
                sa_correct += (sa_pred == labels).sum().item()
                glca_correct += (glca_pred == labels).sum().item()
                combined_correct += (combined_pred == labels).sum().item()
                total_samples += labels.size(0)
        
        results = {
            'sa_accuracy': sa_correct / total_samples * 100,
            'glca_accuracy': glca_correct / total_samples * 100,
            'combined_accuracy': combined_correct / total_samples * 100,
            'total_samples': total_samples
        }
        
        self.logger.info(f"Branch Comparison Results:")
        self.logger.info(f"  SA Branch: {results['sa_accuracy']:.2f}%")
        self.logger.info(f"  GLCA Branch: {results['glca_accuracy']:.2f}%")
        self.logger.info(f"  Combined: {results['combined_accuracy']:.2f}%")
        
        return results
    
    def evaluate_robustness(self, data_loader: DataLoader, 
                           noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3]) -> Dict[str, List[float]]:
        """
        Evaluate model robustness to input noise.
        
        Args:
            data_loader: Data loader for robustness testing
            noise_levels: List of noise standard deviations to test
            
        Returns:
            Dictionary containing robustness results
        """
        self.logger.info(f"Evaluating robustness with noise levels: {noise_levels}")
        
        results = {f'accuracy_noise_{level}': [] for level in noise_levels}
        
        for noise_level in noise_levels:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(data_loader, desc=f"Noise level {noise_level}"):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Add noise
                    if noise_level > 0:
                        noise = torch.randn_like(images) * noise_level
                        images = images + noise
                        images = torch.clamp(images, 0, 1)  # Ensure valid range
                    
                    # Forward pass
                    outputs = self.model(images, x_pair=None)
                    predictions = outputs['sa_logits'].argmax(dim=1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = correct / total * 100
            results[f'accuracy_noise_{noise_level}'] = accuracy
        
        # Log results
        for noise_level in noise_levels:
            acc = results[f'accuracy_noise_{noise_level}']
            self.logger.info(f"  Noise {noise_level}: {acc:.2f}%")
        
        return results