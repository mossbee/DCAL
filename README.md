# Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification

This repository implements the dual cross-attention learning approach proposed in the paper for fine-grained visual categorization (FGVC) and object re-identification (Re-ID) tasks.

## Overview

The implementation includes:
- **Global-Local Cross-Attention (GLCA)**: Enhances interactions between global images and local high-response regions
- **Pair-Wise Cross-Attention (PWCA)**: Regularizes attention learning using image pairs as distractors
- Support for both FGVC and Re-ID tasks with shared architecture
- Integration with timm backbones (ViT, DeiT)

## Project Structure

```
dual_cross_attention_learning/
├── configs/                    # Task-specific configurations
│   ├── __init__.py
│   ├── fgvc/                  # Fine-Grained Visual Categorization configs
│   │   ├── __init__.py
│   │   └── cub200.yaml
│   └── reid/                  # Re-Identification configs
│       ├── __init__.py
│       └── veri776.yaml
├── models/                    # Core model implementations
│   ├── __init__.py
│   ├── attention/             # Attention mechanisms
│   │   ├── __init__.py
│   │   ├── self_attention.py
│   │   ├── glca.py           # Global-Local Cross-Attention
│   │   └── pwca.py           # Pair-Wise Cross-Attention
│   ├── backbones/            # Backbone models
│   │   ├── __init__.py
│   │   └── timm_wrapper.py   # Wrapper for timm models
│   └── dual_attention_model.py # Main model architecture
├── datasets/                  # Dataset handling
│   ├── __init__.py
│   ├── base_dataset.py       # Base dataset class
│   ├── fgvc_datasets.py      # FGVC-specific datasets
│   └── reid_datasets.py      # Re-ID-specific datasets
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── attention_rollout.py  # Attention rollout implementation
│   ├── losses.py             # Loss functions and uncertainty weighting
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py     # Attention visualization
├── training/                  # Training logic
│   ├── __init__.py
│   ├── trainer.py            # Main training loop
│   └── evaluator.py          # Evaluation logic
├── scripts/                   # Training and evaluation scripts
│   ├── train_fgvc.py         # FGVC training script
│   ├── train_reid.py         # Re-ID training script
│   ├── eval_fgvc.py          # FGVC evaluation script
│   └── eval_reid.py          # Re-ID evaluation script
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Usage

### Training FGVC
```bash
python scripts/train_fgvc.py --config configs/fgvc/cub200.yaml
```

### Training Re-ID
```bash
python scripts/train_reid.py --config configs/reid/veri776.yaml
```

## Requirements

See `requirements.txt` for detailed dependencies.