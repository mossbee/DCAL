"""
Backbone model implementations using timm library.

This module provides wrappers for Vision Transformer backbones.
"""

from .timm_wrapper import TimmBackbone

__all__ = ['TimmBackbone']