"""
Evaluation script for Re-Identification (Re-ID) tasks.

This script evaluates trained dual cross-attention models on Re-ID test sets
and computes mAP and CMC metrics.
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


def create_test_datasets(config: dict) -> tuple:
    """
    Create query and gallery datasets based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (query_dataset, gallery_dataset)
    """
    data_config = config.get('data', config)
    
    # Create query dataset
    query_dataset = VeRi776Dataset(
        root_dir=data_config['data_root'],
        split='query',
        transform=None  # Will use default transforms
    )
    
    # Create gallery dataset
    gallery_dataset = VeRi776Dataset(
        root_dir=data_config['data_root'],
        split='gallery',
        transform=None  # Will use default transforms
    )
    
    print(f"Created datasets: Query={len(query_dataset)}, Gallery={len(gallery_dataset)}")
    
    return query_dataset, gallery_dataset


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
    Main evaluation function for Re-ID tasks.
    """
    parser = argparse.ArgumentParser(description='Evaluate dual cross-attention model on Re-ID datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--visualize', action='store_true', help='Generate attention visualizations')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory for results')
    parser.add_argument('--distance_metric', type=str, default='euclidean', choices=['euclidean', 'cosine'], help='Distance metric for evaluation')
    
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
    
    # Create test datasets and data loaders
    query_dataset, gallery_dataset = create_test_datasets(config)
    
    batch_size = data_config.get('batch_size', 64)
    num_workers = data_config.get('num_workers', 4)
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    gallery_loader = DataLoader(
        gallery_dataset,
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
        task_type='reid'
    )
    
    # Evaluate model
    logger.info("Starting Re-ID evaluation...")
    results = evaluator.evaluate_reid(
        query_loader, 
        gallery_loader, 
        distance_metric=args.distance_metric
    )
    
    # Print results
    logger.info("Re-ID Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            if 'mAP' in metric or 'CMC' in metric:
                logger.info(f"  {metric}: {value:.1%}")
            else:
                logger.info(f"  {metric}: {value}")
        else:
            logger.info(f"  {metric}: {value}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'reid_results.json')
    
    import json
    # Convert numpy types for JSON serialization
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            serializable_results[k] = float(v)
        else:
            serializable_results[k] = v
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Generate attention visualizations if requested
    if args.visualize:
        logger.info("Generating attention visualizations...")
        attention_analysis = evaluator.analyze_attention_patterns(
            query_loader,
            num_samples=50,
            save_visualizations=True,
            save_dir=os.path.join(args.output_dir, 'attention_analysis')
        )
        logger.info("Attention analysis completed!")
        
        # Compare attention branches
        branch_comparison = evaluator.compare_branches(query_loader)
        logger.info("Branch comparison:")
        for metric, value in branch_comparison.items():
            logger.info(f"  {metric}: {value:.2f}%")
        
        # Evaluate robustness
        logger.info("Evaluating robustness...")
        robustness_results = evaluator.evaluate_robustness(
            query_loader, 
            noise_levels=[0.0, 0.1, 0.2, 0.3]
        )
        logger.info("Robustness results:")
        for metric, value in robustness_results.items():
            logger.info(f"  {metric}: {value:.2f}%")
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()