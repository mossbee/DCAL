"""
Configuration loading and management utilities.

This module provides utilities for loading YAML configuration files
and managing experiment configurations.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    backbone_name: str = 'deit_small_patch16_224'
    num_classes: int = 200
    embed_dim: int = 384
    num_heads: int = 6
    num_sa_blocks: int = 12
    num_glca_blocks: int = 1
    num_pwca_blocks: int = 12
    top_k_ratio: float = 0.1
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    img_size: int = 224


@dataclass
class DataConfig:
    """Data configuration dataclass."""
    dataset_name: str = 'cub200'
    data_root: str = './data'
    batch_size: int = 32
    num_workers: int = 4
    image_size: list = None
    
    def __post_init__(self):
        if self.image_size is None:
            self.image_size = [224, 224]


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    uncertainty_weighting: bool = True
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    ce_weight: float = 1.0
    
    # Training settings
    log_interval: int = 100
    eval_interval: int = 1
    save_interval: int = 10
    
    # Re-ID specific
    num_instances: int = 4  # For Re-ID batch sampling


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    task_type: str = 'fgvc'  # 'fgvc' or 'reid'
    exp_name: str = 'dual_attention_exp'
    save_dir: str = './experiments'
    device: str = 'cuda'
    seed: int = 42
    
    # Sub-configs
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()


class ConfigManager:
    """Configuration manager for loading and saving configs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig object
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config objects
        config = self._dict_to_config(config_dict)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def save_config(self, config: ExperimentConfig, save_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: ExperimentConfig object to save
            save_path: Path to save YAML file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config_dict = self._config_to_dict(config)
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to: {save_path}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        # Extract main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'data', 'training']}
        
        # Create sub-configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        # Create main config
        config = ExperimentConfig(
            model=model_config,
            data=data_config,
            training=training_config,
            **main_config
        )
        
        return config
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        config_dict = {
            'task_type': config.task_type,
            'exp_name': config.exp_name,
            'save_dir': config.save_dir,
            'device': config.device,
            'seed': config.seed,
            'model': asdict(config.model),
            'data': asdict(config.data),
            'training': asdict(config.training)
        }
        
        return config_dict
    
    def _validate_config(self, config: ExperimentConfig):
        """Validate configuration values."""
        # Validate task type
        if config.task_type not in ['fgvc', 'reid']:
            raise ValueError(f"Invalid task_type: {config.task_type}")
        
        # Validate model parameters
        if config.model.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        
        if config.model.top_k_ratio <= 0 or config.model.top_k_ratio > 1:
            raise ValueError("top_k_ratio must be in (0, 1]")
        
        # Validate training parameters
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if config.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        # Task-specific validation
        if config.task_type == 'reid':
            if config.training.num_instances < 2:
                raise ValueError("num_instances must be >= 2 for Re-ID")
        
        self.logger.info("Configuration validation passed")
    
    def update_config_for_dataset(self, config: ExperimentConfig, 
                                dataset_name: str) -> ExperimentConfig:
        """
        Update configuration based on dataset.
        
        Args:
            config: Base configuration
            dataset_name: Name of the dataset
            
        Returns:
            Updated configuration
        """
        if dataset_name.lower() == 'cub200':
            config.model.num_classes = 200
            config.data.image_size = [448, 448]
            config.task_type = 'fgvc'
            
        elif dataset_name.lower() == 'veri776':
            config.model.num_classes = 576  # Training identities
            config.data.image_size = [256, 256]
            config.task_type = 'reid'
            config.training.num_instances = 4
            
        else:
            self.logger.warning(f"Unknown dataset: {dataset_name}")
        
        return config
    
    def merge_configs(self, base_config: ExperimentConfig, 
                     override_config: Dict[str, Any]) -> ExperimentConfig:
        """
        Merge base configuration with override values.
        
        Args:
            base_config: Base configuration
            override_config: Override values
            
        Returns:
            Merged configuration
        """
        # Convert base config to dict
        config_dict = self._config_to_dict(base_config)
        
        # Deep merge override values
        self._deep_merge(config_dict, override_config)
        
        # Convert back to config object
        return self._dict_to_config(config_dict)
    
    def _deep_merge(self, base_dict: Dict[str, Any], 
                   override_dict: Dict[str, Any]):
        """Deep merge two dictionaries."""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value


def load_config(config_path: str) -> ExperimentConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ExperimentConfig object
    """
    manager = ConfigManager()
    return manager.load_config(config_path)


def save_config(config: ExperimentConfig, save_path: str):
    """
    Convenience function to save configuration.
    
    Args:
        config: Configuration to save
        save_path: Path to save file
    """
    manager = ConfigManager()
    manager.save_config(config, save_path)


def create_default_config(task_type: str = 'fgvc', 
                         dataset_name: str = 'cub200') -> ExperimentConfig:
    """
    Create default configuration for a task.
    
    Args:
        task_type: Task type ('fgvc' or 'reid')
        dataset_name: Dataset name
        
    Returns:
        Default ExperimentConfig
    """
    config = ExperimentConfig(task_type=task_type)
    
    manager = ConfigManager()
    config = manager.update_config_for_dataset(config, dataset_name)
    
    return config


# Example usage and config templates
FGVC_CONFIG_TEMPLATE = {
    'task_type': 'fgvc',
    'exp_name': 'cub200_dual_attention',
    'save_dir': './experiments/fgvc',
    'device': 'cuda',
    'seed': 42,
    
    'model': {
        'backbone_name': 'deit_small_patch16_224',
        'num_classes': 200,
        'embed_dim': 384,
        'num_heads': 6,
        'top_k_ratio': 0.1,
        'drop_path_rate': 0.1
    },
    
    'data': {
        'dataset_name': 'cub200',
        'data_root': './data/CUB_200_2011',
        'batch_size': 32,
        'image_size': [448, 448]
    },
    
    'training': {
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5
    }
}

REID_CONFIG_TEMPLATE = {
    'task_type': 'reid',
    'exp_name': 'veri776_dual_attention',
    'save_dir': './experiments/reid',
    'device': 'cuda',
    'seed': 42,
    
    'model': {
        'backbone_name': 'deit_small_patch16_224',
        'num_classes': 576,
        'embed_dim': 384,
        'num_heads': 6,
        'top_k_ratio': 0.3,
        'drop_path_rate': 0.1
    },
    
    'data': {
        'dataset_name': 'veri776',
        'data_root': './data/VeRi',
        'batch_size': 64,
        'num_instances': 4,
        'image_size': [256, 256]
    },
    
    'training': {
        'num_epochs': 120,
        'learning_rate': 8e-3,
        'optimizer': 'sgd',
        'scheduler': 'multistep',
        'triplet_margin': 0.3
    }
}