"""
Evaluation script for Fine-Grained Visual Categorization (FGVC) tasks.

This script evaluates trained dual cross-attention models on FGVC test sets
and computes accuracy metrics.
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
from datasets import CUB200Dataset
from training import DualAttentionEvaluator
from utils import load_config as load_yaml_config


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


def create_test_dataset(config: dict):
    """
    Create test dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Test dataset
    """
    data_config = config.get('data', config)
    
    # Create test dataset
    test_dataset = CUB200Dataset(
        root_dir=data_config['data_root'],
        split='test',
        transform=None  # Will use default transforms
    )
    
    print(f"Created test dataset: {len(test_dataset)} samples")
    
    return test_dataset


def load_model(checkpoint_path: str, config: dict) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        
    Returns:
        Loaded model
    """
    model_config = config.get('model', config)
    
    # Create model
    model = DualAttentionModel(
        backbone_name=model_config.get('backbone_name', 'deit_small_patch16_224'),
        num_classes=model_config.get('num_classes', 200),
        task_type='fgvc',
        num_sa_blocks=model_config.get('num_sa_blocks', 12),
        num_glca_blocks=model_config.get('num_glca_blocks', 1),
        num_pwca_blocks=model_config.get('num_pwca_blocks', 12),
        embed_dim=model_config.get('embed_dim', 384),
        num_heads=model_config.get('num_heads', 6),
        top_k_ratio=model_config.get('top_k_ratio', 0.1),
        drop_rate=model_config.get('drop_rate', 0.0),
        drop_path_rate=model_config.get('drop_path_rate', 0.1),
        img_size=model_config.get('img_size', 224)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model from: {checkpoint_path}")
    
    return model


def main():
    """
    Main evaluation function for FGVC tasks.
    """
    parser = argparse.ArgumentParser(description='Evaluate dual cross-attention model on FGVC datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--visualize', action='store_true', help='Generate attention visualizations')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    data_config = config.get('data', config)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create test dataset and data loader
    test_dataset = create_test_dataset(config)
    
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Load model
    model = load_model(args.checkpoint, config)
    model.to(device)
    
    # Create evaluator
    evaluator = DualAttentionEvaluator(
        model=model,
        device=device,
        task_type='fgvc'
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_fgvc(test_loader, compute_per_class=True)
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, dict):
            logger.info(f"{metric}: [Dict with {len(value)} entries]")
        elif isinstance(value, (int, float)):
            if 'accuracy' in metric.lower():
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'fgvc_results.json')
    
    import json
    # Convert numpy types for JSON serialization
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable_results[k] = {str(kk): float(vv) for kk, vv in v.items()}
        elif isinstance(v, (list, tuple)):
            serializable_results[k] = [float(x) for x in v]
        else:
            serializable_results[k] = float(v) if isinstance(v, (int, float)) else v
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Generate attention visualizations if requested
    if args.visualize:
        logger.info("Generating attention visualizations...")
        attention_analysis = evaluator.analyze_attention_patterns(
            test_loader, 
            num_samples=50,
            save_visualizations=True,
            save_dir=os.path.join(args.output_dir, 'attention_analysis')
        )
        logger.info("Attention analysis completed!")
        
        # Compare attention branches
        branch_comparison = evaluator.compare_branches(test_loader)
        logger.info("Branch comparison:")
        for metric, value in branch_comparison.items():
            logger.info(f"  {metric}: {value:.2f}%")
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()