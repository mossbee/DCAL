"""
Training and evaluation modules for dual cross-attention learning.

This module contains the main training loop and evaluation logic
for both FGVC and Re-ID tasks.
"""

from .trainer import DualAttentionTrainer
from .evaluator import DualAttentionEvaluator

__all__ = ['DualAttentionTrainer', 'DualAttentionEvaluator']