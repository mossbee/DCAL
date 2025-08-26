"""
Training script for Re-Identification (Re-ID) tasks.

This script handles training the dual cross-attention model on Re-ID datasets
including VeRi-776.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DualAttentionModel
from datasets import VeRi776Dataset
from training import DualAttentionTrainer
from utils import UncertaintyWeightedLoss, TripletLoss, load_config as load_yaml_config


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


def create_dataset(config: dict) -> tuple:
    """
    Create train and validation datasets based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_config = config.get('data', config)
    
    # Create training dataset
    train_dataset = VeRi776Dataset(
        root_dir=data_config['data_root'],
        split='train',
        transform=None  # Will use default transforms
    )
    
    # Create validation dataset (using gallery split for now)
    val_dataset = VeRi776Dataset(
        root_dir=data_config['data_root'],
        split='gallery',
        transform=None  # Will use default transforms
    )
    
    print(f"Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_model(config: dict) -> nn.Module:
    """
    Create dual attention model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', config)
    
    model = DualAttentionModel(
        backbone_name=model_config.get('backbone_name', 'deit_small_patch16_224'),
        num_classes=model_config.get('num_classes', 576),
        task_type='reid',
        num_sa_blocks=model_config.get('num_sa_blocks', 12),
        num_glca_blocks=model_config.get('num_glca_blocks', 1),
        num_pwca_blocks=model_config.get('num_pwca_blocks', 12),
        embed_dim=model_config.get('embed_dim', 384),
        num_heads=model_config.get('num_heads', 6),
        top_k_ratio=model_config.get('top_k_ratio', 0.3),
        drop_rate=model_config.get('drop_rate', 0.0),
        drop_path_rate=model_config.get('drop_path_rate', 0.1),
        img_size=model_config.get('img_size', 224)
    )
    
    print(f"Created model: {model_config.get('backbone_name', 'deit_small_patch16_224')} with {model_config.get('num_classes', 576)} classes")
    
    return model


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Configured optimizer
    """
    training_config = config.get('training', config)
    
    optimizer_name = training_config.get('optimizer', 'sgd').lower()
    learning_rate = training_config.get('learning_rate', 8e-3)
    weight_decay = training_config.get('weight_decay', 1e-4)
    
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    elif optimizer_name == 'adamw':
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
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"Created optimizer: {optimizer_name} with lr={learning_rate}")
    
    return optimizer


def main():
    """
    Main training function for Re-ID tasks.
    """
    parser = argparse.ArgumentParser(description='Train dual cross-attention model on Re-ID datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    data_config = config.get('data', config)
    training_config = config.get('training', config)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets and data loaders
    train_dataset, val_dataset = create_dataset(config)
    
    # Create data loaders with Re-ID specific settings
    batch_size = data_config.get('batch_size', 64)
    num_workers = data_config.get('num_workers', 4)
    num_instances = training_config.get('num_instances', 4)
    
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
    scheduler_type = training_config.get('scheduler', 'multistep')
    num_epochs = training_config.get('num_epochs', 120)
    
    if scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=training_config.get('milestones', [40, 80]),
            gamma=training_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=training_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer_kwargs = {
        'log_interval': training_config.get('log_interval', 100),
        'eval_interval': training_config.get('eval_interval', 1),
        'save_interval': training_config.get('save_interval', 10)
    }
    
    trainer = DualAttentionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task_type='reid',
        **trainer_kwargs
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from: {args.resume}")
    
    # Start training
    save_dir = config.get('save_dir', './experiments/reid')
    history = trainer.train(
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    logger.info("Training completed!")
    logger.info(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()