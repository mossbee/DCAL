"""
Main training loop for dual cross-attention learning.

This module implements the training logic with support for both
FGVC and Re-ID tasks, including pair sampling for PWCA.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import logging
from tqdm import tqdm
import os
import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.losses import UncertaintyWeightedLoss, TripletLoss, CombinedLoss
from utils.metrics import compute_accuracy, MetricTracker
from models.attention.pwca import PairSampler


class DualAttentionTrainer:
    """
    Main trainer for dual cross-attention model.
    
    Handles training loop with uncertainty-weighted loss, pair sampling
    for PWCA, and evaluation during training.
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: Optional[DataLoader] = None, optimizer: torch.optim.Optimizer = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda', task_type: str = 'fgvc', 
                 log_interval: int = 100, eval_interval: int = 1, save_interval: int = 10,
                 **kwargs):
        """
        Initialize dual attention trainer.
        
        Args:
            model: Dual attention model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device for training ('cuda' or 'cpu')
            task_type: Task type ('fgvc' or 'reid')
            log_interval: Steps between logging
            eval_interval: Epochs between evaluation
            save_interval: Epochs between saving checkpoints
            **kwargs: Additional training arguments
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.task_type = task_type
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Initialize loss functions
        self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=3)
        
        if task_type == 'fgvc':
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == 'reid':
            self.criterion = CombinedLoss(
                ce_weight=1.0, 
                triplet_weight=1.0, 
                triplet_margin=0.3,
                num_classes=model.num_classes
            )
        
        # Initialize pair sampler for PWCA
        self.pair_sampler = PairSampler(strategy='random')
        
        # Initialize metric tracking
        if task_type == 'fgvc':
            metric_names = ['loss', 'sa_loss', 'glca_loss', 'pwca_loss', 'accuracy', 'top5_accuracy']
        else:
            metric_names = ['loss', 'sa_loss', 'glca_loss', 'pwca_loss', 'ce_loss', 'triplet_loss']
        
        self.train_metrics = MetricTracker(metric_names)
        self.val_metrics = MetricTracker(metric_names)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for step, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Sample pairs for PWCA
            batch_with_pairs = self._sample_pairs(batch)
            paired_images = batch_with_pairs.get('paired_images')
            if paired_images is not None:
                paired_images = paired_images.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, paired_images)
            
            # Compute losses
            loss_dict = self._compute_loss(outputs, labels, batch)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = images.size(0)
            metrics_update = {
                'loss': total_loss.item(),
                'sa_loss': loss_dict.get('sa_loss', 0),
                'glca_loss': loss_dict.get('glca_loss', 0),
                'pwca_loss': loss_dict.get('pwca_loss', 0)
            }
            
            # Task-specific metrics
            if self.task_type == 'fgvc':
                sa_logits = outputs['sa_logits']
                acc1, acc5 = compute_accuracy(sa_logits, labels, topk=(1, 5))
                metrics_update.update({
                    'accuracy': acc1,
                    'top5_accuracy': acc5
                })
            elif self.task_type == 'reid':
                metrics_update.update({
                    'ce_loss': loss_dict.get('ce_loss', 0),
                    'triplet_loss': loss_dict.get('triplet_loss', 0)
                })
            
            self.train_metrics.update(metrics_update, batch_size)
            
            # Update progress bar
            if step % self.log_interval == 0:
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        return self.train_metrics.get_averages()
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass (no PWCA during validation)
                outputs = self.model(images, x_pair=None)
                
                # Compute losses
                loss_dict = self._compute_loss(outputs, labels, batch)
                total_loss = loss_dict['total_loss']
                
                # Update metrics
                batch_size = images.size(0)
                metrics_update = {
                    'loss': total_loss.item(),
                    'sa_loss': loss_dict.get('sa_loss', 0),
                    'glca_loss': loss_dict.get('glca_loss', 0)
                }
                
                # Task-specific metrics
                if self.task_type == 'fgvc':
                    sa_logits = outputs['sa_logits']
                    acc1, acc5 = compute_accuracy(sa_logits, labels, topk=(1, 5))
                    metrics_update.update({
                        'accuracy': acc1,
                        'top5_accuracy': acc5
                    })
                elif self.task_type == 'reid':
                    metrics_update.update({
                        'ce_loss': loss_dict.get('ce_loss', 0),
                        'triplet_loss': loss_dict.get('triplet_loss', 0)
                    })
                
                self.val_metrics.update(metrics_update, batch_size)
        
        return self.val_metrics.get_averages()
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints') -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_metric = 0.0
        start_time = time.time()
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Task: {self.task_type.upper()}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = {}
            if epoch % self.eval_interval == 0:
                val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics.get('loss', 0))
                if self.task_type == 'fgvc':
                    self.history['train_acc'].append(train_metrics.get('accuracy', 0))
                    self.history['val_acc'].append(val_metrics.get('accuracy', 0))
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_metrics.get('loss', train_metrics.get('loss', 0))
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start
            log_str = f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | "
            log_str += f"Train: {self.train_metrics.summary()}"
            if val_metrics:
                log_str += f" | Val: {self.val_metrics.summary()}"
            
            # Log uncertainty weights
            uncertainty_weights = self.uncertainty_loss.get_current_weights()
            weight_str = " | ".join([f"{k}: {v:.3f}" for k, v in uncertainty_weights.items()])
            log_str += f" | Weights: {weight_str}"
            
            self.logger.info(log_str)
            
            # Save checkpoint
            current_metric = val_metrics.get('accuracy', train_metrics.get('accuracy', 0))
            is_best = current_metric > best_metric
            
            if is_best:
                best_metric = current_metric
            
            if epoch % self.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, save_dir, 
                                   {**train_metrics, **val_metrics}, 
                                   is_best=is_best)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.1f} hours")
        self.logger.info(f"Best metric: {best_metric:.4f}")
        
        return self.history
    
    def _sample_pairs(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Sample image pairs for PWCA training.
        
        Args:
            batch_data: Batch of training data
            
        Returns:
            Batch data with paired images
        """
        # Convert batch format for pair sampler
        sampler_input = {
            'images': batch_data['image'],
            'labels': batch_data['label']
        }
        
        # Sample pairs
        paired_batch = self.pair_sampler.sample_pairs(sampler_input)
        
        # Add paired images to original batch
        batch_with_pairs = batch_data.copy()
        batch_with_pairs['paired_images'] = paired_batch['paired_images']
        batch_with_pairs['paired_labels'] = paired_batch['paired_labels']
        
        return batch_with_pairs
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with uncertainty weighting.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth targets
            batch: Full batch data
            
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        
        # Self-Attention loss
        sa_logits = outputs['sa_logits']
        if self.task_type == 'fgvc':
            sa_loss = self.criterion(sa_logits, targets)
        else:
            sa_features = outputs['sa_features'][:, 0]  # CLS token
            sa_loss_dict = self.criterion(sa_logits, sa_features, targets)
            sa_loss = sa_loss_dict['total_loss']
            losses.update({
                'ce_loss': sa_loss_dict['ce_loss'].item(),
                'triplet_loss': sa_loss_dict['triplet_loss'].item()
            })
        
        losses['sa_loss'] = sa_loss
        
        # Global-Local Cross-Attention loss
        glca_logits = outputs['glca_logits']
        if self.task_type == 'fgvc':
            glca_loss = self.criterion(glca_logits, targets)
        else:
            glca_features = outputs['glca_features'][:, 0]  # CLS token
            glca_loss_dict = self.criterion(glca_logits, glca_features, targets)
            glca_loss = glca_loss_dict['total_loss']
        
        losses['glca_loss'] = glca_loss
        
        # Pair-Wise Cross-Attention loss (only during training)
        pwca_loss = None
        if 'pwca_logits' in outputs:
            pwca_logits = outputs['pwca_logits']
            if self.task_type == 'fgvc':
                pwca_loss = self.criterion(pwca_logits, targets)
            else:
                pwca_features = outputs['pwca_features'][:, 0]  # CLS token
                pwca_loss_dict = self.criterion(pwca_logits, pwca_features, targets)
                pwca_loss = pwca_loss_dict['total_loss']
            
            losses['pwca_loss'] = pwca_loss
        
        # Uncertainty-weighted total loss
        loss_dict_for_weighting = {
            'sa_loss': losses['sa_loss'],
            'glca_loss': losses['glca_loss']
        }
        
        if pwca_loss is not None:
            loss_dict_for_weighting['pwca_loss'] = pwca_loss
        
        weighted_loss_dict = self.uncertainty_loss(loss_dict_for_weighting)
        
        # Combine all losses
        result = {
            'total_loss': weighted_loss_dict['total_loss'],
            'sa_loss': losses['sa_loss'].item(),
            'glca_loss': losses['glca_loss'].item(),
        }
        
        if pwca_loss is not None:
            result['pwca_loss'] = pwca_loss.item()
        
        if 'ce_loss' in losses:
            result['ce_loss'] = losses['ce_loss']
            result['triplet_loss'] = losses['triplet_loss']
        
        return result
    
    def save_checkpoint(self, epoch: int, save_dir: str, 
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            save_dir: Directory to save checkpoint
            metrics: Optional metrics to save with checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'uncertainty_loss_state_dict': self.uncertainty_loss.state_dict(),
            'history': self.history,
            'metrics': metrics or {}
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch+1}")
        
        # Save training history as JSON
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'uncertainty_loss_state_dict' in checkpoint:
            self.uncertainty_loss.load_state_dict(checkpoint['uncertainty_loss_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Loaded checkpoint from epoch {epoch+1}")
        
        return epoch