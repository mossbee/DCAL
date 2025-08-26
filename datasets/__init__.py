"""
Dataset implementations for FGVC and Re-ID tasks.

This module provides dataset classes and data loaders for fine-grained
visual categorization and re-identification tasks.
"""

from .fgvc_datasets import CUB200Dataset
from .reid_datasets import VeRi776Dataset
from .base_dataset import BaseDataset

__all__ = [
    'BaseDataset',
    'CUB200Dataset', 'VeRi776Dataset'
]