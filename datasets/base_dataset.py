"""
Base dataset class for common functionality.

This module provides the base dataset class with common preprocessing
and augmentation methods shared across FGVC and Re-ID tasks.
"""

import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple, List
from PIL import Image
import torchvision.transforms as transforms
import os


class BaseDataset(Dataset):
    """
    Base dataset class with common functionality.
    
    Provides common methods for data loading, preprocessing, and augmentation
    that are shared between FGVC and Re-ID datasets.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None,
                 task_type: str = 'fgvc'):
        """
        Initialize base dataset.
        
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Torchvision transforms to apply
            task_type: Type of task ('fgvc' or 'reid')
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.task_type = task_type
        
        # Set default transforms if none provided
        if transform is None:
            self.transform = self.get_default_transforms(split)
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Number of samples in the dataset
        """
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'image': Preprocessed image tensor
            - 'label': Class label or identity ID
            - 'path': Image file path
            - Additional task-specific fields
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")
    
    def get_default_transforms(self, split: str) -> transforms.Compose:
        """
        Get default transforms for the task and split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Composed transforms
        """
        if self.task_type == 'fgvc':
            return self._get_fgvc_transforms(split)
        elif self.task_type == 'reid':
            return self._get_reid_transforms(split)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _get_fgvc_transforms(self, split: str) -> transforms.Compose:
        """
        Get transforms for FGVC tasks (following paper settings).
        
        For training: Resize to 550x550, random crop to 448x448
        For testing: Resize to 448x448
        """
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((550, 550)),
                transforms.RandomCrop((448, 448)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _get_reid_transforms(self, split: str) -> transforms.Compose:
        """
        Get transforms for Re-ID tasks (following paper settings).
        
        For person Re-ID: 256x128
        For vehicle Re-ID: 256x256
        """
        # Default to person Re-ID size, can be overridden by subclasses
        img_size = (256, 128)
        
        if split == 'train':
            return transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                transforms.RandomCrop(img_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        
        Returns:
            Tensor of class weights
        """
        if not hasattr(self, 'samples'):
            raise NotImplementedError("Dataset must have 'samples' attribute")
        
        # Count samples per class
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Compute inverse frequency weights
        total_samples = len(self.samples)
        num_classes = len(class_counts)
        
        weights = torch.zeros(num_classes)
        for class_id, count in class_counts.items():
            weights[class_id] = total_samples / (num_classes * count)
        
        return weights
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'num_samples': len(self),
            'split': self.split,
            'task_type': self.task_type,
            'root_dir': self.root_dir
        }
        
        if hasattr(self, 'samples'):
            # Class distribution
            class_counts = {}
            for _, label in self.samples:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            stats.update({
                'num_classes': len(class_counts),
                'min_samples_per_class': min(class_counts.values()),
                'max_samples_per_class': max(class_counts.values()),
                'avg_samples_per_class': sum(class_counts.values()) / len(class_counts)
            })
        
        return stats