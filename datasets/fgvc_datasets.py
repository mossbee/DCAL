"""
Fine-Grained Visual Categorization (FGVC) dataset implementations.

This module implements dataset classes for CUB-200-2011 dataset.
"""

import os
from typing import Dict, Any, List, Tuple
import pandas as pd
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.base_dataset import BaseDataset


class CUB200Dataset(BaseDataset):
    """
    CUB-200-2011 bird species dataset.
    
    Contains 11,788 images of 200 bird species with 5,994 for training
    and 5,794 for testing.
    
    Dataset structure:
    CUB_200_2011/
    ├── images.txt              # <image_id> <image_name>
    ├── train_test_split.txt    # <image_id> <is_training_image>
    ├── classes.txt             # <class_id> <class_name>
    ├── image_class_labels.txt  # <image_id> <class_id>
    └── images/                 # Image files organized by species
        ├── 001.Black_footed_Albatross/
        ├── 002.Laysan_Albatross/
        └── ...
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Initialize CUB-200 dataset.
        
        Args:
            root_dir: Root directory containing CUB_200_2011 folder
            split: Dataset split ('train' or 'test')
            transform: Torchvision transforms to apply
        """
        super().__init__(root_dir, split, transform, task_type='fgvc')
        
        # Set up paths
        self.cub_root = os.path.join(root_dir, 'CUB_200_2011')
        self.images_dir = os.path.join(self.cub_root, 'images')
        
        # Verify dataset exists
        if not os.path.exists(self.cub_root):
            raise FileNotFoundError(f"CUB-200-2011 dataset not found at {self.cub_root}")
        
        # Load annotations
        self.samples = self._load_annotations()
        
        # Load class names for reference
        self.class_names = self._load_class_names()
        
        print(f"CUB-200 {split} set: {len(self.samples)} images, {len(self.class_names)} classes")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'image': Preprocessed image tensor
            - 'label': Class label (0-199)
            - 'image_id': Original image ID
            - 'image_path': Path to image file
            - 'class_name': Human-readable class name
        """
        image_path, label = self.samples[idx]
        
        # Load image
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'image_id': idx,
            'image_path': image_path,
            'class_name': self.class_names[label]
        }
    
    def _load_annotations(self) -> List[Tuple[str, int]]:
        """
        Load image paths and labels from CUB annotation files.
        
        Returns:
            List of (image_path, class_id) tuples
        """
        # Load image list: <image_id> <image_name>
        images_file = os.path.join(self.cub_root, 'images.txt')
        image_names = {}
        with open(images_file, 'r') as f:
            for line in f:
                image_id, image_name = line.strip().split(' ', 1)
                image_names[int(image_id)] = image_name
        
        # Load train/test split: <image_id> <is_training_image>
        split_file = os.path.join(self.cub_root, 'train_test_split.txt')
        train_test_split = {}
        with open(split_file, 'r') as f:
            for line in f:
                image_id, is_training = line.strip().split()
                train_test_split[int(image_id)] = int(is_training)
        
        # Load class labels: <image_id> <class_id>
        labels_file = os.path.join(self.cub_root, 'image_class_labels.txt')
        image_labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                image_id, class_id = line.strip().split()
                image_labels[int(image_id)] = int(class_id) - 1  # Convert to 0-based indexing
        
        # Filter by split and create sample list
        samples = []
        target_split = 1 if self.split == 'train' else 0
        
        for image_id in image_names:
            if train_test_split[image_id] == target_split:
                image_path = os.path.join(self.images_dir, image_names[image_id])
                label = image_labels[image_id]
                samples.append((image_path, label))
        
        return samples
    
    def _load_class_names(self) -> List[str]:
        """
        Load class names from classes.txt.
        
        Returns:
            List of class names indexed by class_id (0-based)
        """
        classes_file = os.path.join(self.cub_root, 'classes.txt')
        class_names = [''] * 200  # Pre-allocate for 200 classes
        
        with open(classes_file, 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split(' ', 1)
                class_names[int(class_id) - 1] = class_name  # Convert to 0-based indexing
        
        return class_names
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution for the current split.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        class_counts = {}
        for _, label in self.samples:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts
    
    def get_sample_by_class(self, class_name: str, max_samples: int = 10) -> List[Tuple[str, int]]:
        """
        Get sample images for a specific class.
        
        Args:
            class_name: Name of the bird species
            max_samples: Maximum number of samples to return
            
        Returns:
            List of (image_path, label) tuples
        """
        if class_name not in self.class_names:
            raise ValueError(f"Class {class_name} not found in dataset")
        
        target_label = self.class_names.index(class_name)
        class_samples = [(path, label) for path, label in self.samples if label == target_label]
        
        return class_samples[:max_samples]