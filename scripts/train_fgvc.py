"""
Training script for Fine-Grained Visual Categorization (FGVC) tasks.

This script handles training the dual cross-attention model on FGVC datasets
including CUB-200.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import logging

# Add the project root to Python path to avoid conflicts with system packages
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dual_attention_model import DualAttentionModel
from datasets.fgvc_datasets import CUB200Dataset
from training.trainer import DualAttentionTrainer
from utils.losses import UncertaintyWeightedLoss
from utils.config import load_config as load_yaml_config


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config = load_yaml_config(config_path)
        return config.__dict__ if hasattr(config, '__dict__') else config
    except:
        # Fallback to direct YAML loading
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def create_dataset(config) -> tuple:
    """
    Create train and validation datasets based on configuration.
    
    Args:
        config: Configuration object (ExperimentConfig)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Handle both dict and dataclass config formats
    if hasattr(config, 'data'):
        data_config = config.data
    else:
        data_config = config.get('data', config)
    
    # Extract data_root with proper attribute/key access
    if hasattr(data_config, 'data_root'):
        data_root = data_config.data_root
    else:
        data_root = data_config['data_root']
    
    # Create training dataset
    train_dataset = CUB200Dataset(
        root_dir=data_root,
        split='train',
        transform=None  # Will use default transforms
    )
    
    # Create validation dataset (using test split for now)
    val_dataset = CUB200Dataset(
        root_dir=data_root,
        split='test',
        transform=None  # Will use default transforms
    )
    
    print(f"Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_model(config) -> nn.Module:
    """
    Create dual attention model based on configuration.
    
    Args:
        config: Configuration object (ExperimentConfig)
        
    Returns:
        Initialized model
    """
    # Handle both dict and dataclass config formats
    if hasattr(config, 'model'):
        model_config = config.model
    else:
        model_config = config.get('model', config)
    
    # Helper function to get attribute or key value
    def get_value(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        else:
            return default
    
    model = DualAttentionModel(
        backbone_name=get_value(model_config, 'backbone_name', 'deit_small_patch16_224'),
        num_classes=get_value(model_config, 'num_classes', 200),
        task_type='fgvc',
        num_sa_blocks=get_value(model_config, 'num_sa_blocks', 12),
        num_glca_blocks=get_value(model_config, 'num_glca_blocks', 1),
        num_pwca_blocks=get_value(model_config, 'num_pwca_blocks', 12),
        embed_dim=get_value(model_config, 'embed_dim', 384),
        num_heads=get_value(model_config, 'num_heads', 6),
        top_k_ratio=get_value(model_config, 'top_k_ratio', 0.1),
        drop_rate=get_value(model_config, 'drop_rate', 0.0),
        drop_path_rate=get_value(model_config, 'drop_path_rate', 0.1),
        img_size=get_value(model_config, 'img_size', 224)
    )
    
    backbone_name = get_value(model_config, 'backbone_name', 'deit_small_patch16_224')
    num_classes = get_value(model_config, 'num_classes', 200)
    print(f"Created model: {backbone_name} with {num_classes} classes")
    
    return model


def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration object (ExperimentConfig)
        
    Returns:
        Configured optimizer
    """
    # Handle both dict and dataclass config formats
    if hasattr(config, 'training'):
        training_config = config.training
    else:
        training_config = config.get('training', config)
    
    # Helper function to get attribute or key value
    def get_value(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        else:
            return default
    
    optimizer_name = get_value(training_config, 'optimizer', 'adamw').lower()
    learning_rate = get_value(training_config, 'learning_rate', 1e-4)
    weight_decay = get_value(training_config, 'weight_decay', 1e-4)
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"Created optimizer: {optimizer_name} with lr={learning_rate}")
    
    return optimizer


def main():
    """
    Main training function for FGVC tasks.
    """
    parser = argparse.ArgumentParser(description='Train dual cross-attention model on FGVC datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle both dict and dataclass config formats
    if hasattr(config, 'data'):
        data_config = config.data
        training_config = config.training
    else:
        data_config = config.get('data', config)
        training_config = config.get('training', config)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets and data loaders
    train_dataset, val_dataset = create_dataset(config)
    
    # Create data loaders with default settings
    def get_value(obj, key, default):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        else:
            return default
    
    batch_size = get_value(data_config, 'batch_size', 32)
    num_workers = get_value(data_config, 'num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler_type = get_value(training_config, 'scheduler', 'cosine')
    num_epochs = get_value(training_config, 'num_epochs', 100)
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=get_value(training_config, 'min_lr', 1e-6)
        )
    elif scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=get_value(training_config, 'milestones', [30, 60, 90]),
            gamma=get_value(training_config, 'gamma', 0.1)
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer_kwargs = {
        'log_interval': get_value(training_config, 'log_interval', 100),
        'eval_interval': get_value(training_config, 'eval_interval', 1),
        'save_interval': get_value(training_config, 'save_interval', 10)
    }
    
    trainer = DualAttentionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task_type='fgvc',
        **trainer_kwargs
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from: {args.resume}")
    
    # Start training
    save_dir = get_value(config, 'save_dir', './experiments/fgvc')
    history = trainer.train(
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    logger.info("Training completed!")
    logger.info(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()