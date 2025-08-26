"""
Re-Identification (Re-ID) dataset implementations.

This module implements dataset classes for VeRi-776 dataset.
"""

import os
import glob
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.base_dataset import BaseDataset


class VeRi776Dataset(BaseDataset):
    """
    VeRi-776 vehicle re-identification dataset.
    
    Contains 51,035 images of 776 vehicles from 20 cameras.
    - 576 vehicles for training (37,778 images)
    - 200 vehicles for testing (11,579 gallery + 1,678 query images)
    
    Dataset structure:
    VeRi/
    ├── image_train/           # Training images
    ├── image_test/            # Gallery images  
    ├── image_query/           # Query images
    ├── name_train.txt         # Training image names
    ├── name_test.txt          # Gallery image names
    ├── name_query.txt         # Query image names
    ├── train_label.xml        # Training labels (vehicle_id, camera_id, etc.)
    ├── test_label.xml         # Test labels
    └── gt_image.txt           # Ground truth for queries
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Initialize VeRi-776 dataset.
        
        Args:
            root_dir: Root directory containing VeRi data
            split: Dataset split ('train', 'query', 'gallery')
            transform: Torchvision transforms to apply
        """
        # Override default transforms for vehicle Re-ID (square images)
        super().__init__(root_dir, split, transform, task_type='reid')
        
        # Set up paths
        self.veri_root = os.path.join(root_dir, 'VeRi')
        
        # Verify dataset exists
        if not os.path.exists(self.veri_root):
            raise FileNotFoundError(f"VeRi-776 dataset not found at {self.veri_root}")
        
        # Load annotations
        self.samples = self._load_annotations()
        
        print(f"VeRi-776 {split} set: {len(self.samples)} images")
    
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
            - 'vehicle_id': Vehicle identity ID
            - 'camera_id': Camera ID
            - 'image_path': Path to image file
            - 'image_name': Image filename
        """
        image_path, vehicle_id, camera_id, image_name = self.samples[idx]
        
        # Load image
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'vehicle_id': vehicle_id,
            'camera_id': camera_id,
            'label': vehicle_id,  # For compatibility with training code
            'image_path': image_path,
            'image_name': image_name
        }
    
    def _load_annotations(self) -> List[Tuple[str, int, int, str]]:
        """
        Load image paths, vehicle IDs, and camera IDs.
        
        Returns:
            List of (image_path, vehicle_id, camera_id, image_name) tuples
        """
        samples = []
        
        if self.split == 'train':
            # Load training annotations from XML
            xml_file = os.path.join(self.veri_root, 'train_label.xml')
            image_dir = os.path.join(self.veri_root, 'image_train')
            samples = self._parse_xml_labels(xml_file, image_dir)
            
        elif self.split == 'gallery':
            # Load test/gallery annotations from XML
            xml_file = os.path.join(self.veri_root, 'test_label.xml')
            image_dir = os.path.join(self.veri_root, 'image_test')
            samples = self._parse_xml_labels(xml_file, image_dir)
            
        elif self.split == 'query':
            # Load query annotations
            # Query images use the same labels as test but from query directory
            xml_file = os.path.join(self.veri_root, 'test_label.xml')
            image_dir = os.path.join(self.veri_root, 'image_query')
            
            # Load query image names
            query_names_file = os.path.join(self.veri_root, 'name_query.txt')
            query_names = set()
            if os.path.exists(query_names_file):
                with open(query_names_file, 'r') as f:
                    for line in f:
                        query_names.add(line.strip())
            
            # Parse XML and filter for query images
            all_samples = self._parse_xml_labels(xml_file, image_dir)
            samples = [(path, vid, cid, name) for path, vid, cid, name in all_samples 
                      if name in query_names and os.path.exists(path)]
        
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        return samples
    
    def _parse_xml_labels(self, xml_file: str, image_dir: str) -> List[Tuple[str, int, int, str]]:
        """
        Parse XML label file to extract vehicle and camera IDs.
        
        Args:
            xml_file: Path to XML label file
            image_dir: Directory containing images
            
        Returns:
            List of (image_path, vehicle_id, camera_id, image_name) tuples
        """
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"Label file not found: {xml_file}")
        
        samples = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for item in root.findall('.//Item'):
            # Extract attributes
            image_name = item.get('imageName')
            vehicle_id = int(item.get('vehicleID'))
            camera_id = int(item.get('cameraID'))
            
            # Build full image path
            image_path = os.path.join(image_dir, image_name)
            
            # Only include if image exists
            if os.path.exists(image_path):
                samples.append((image_path, vehicle_id, camera_id, image_name))
        
        return samples
    
    def _get_reid_transforms(self, split: str):
        """Override to use square images for vehicle Re-ID."""
        import torchvision.transforms as transforms
        
        # VeRi uses 256x256 square images for vehicles
        img_size = (256, 256)
        
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
    
    def get_num_vehicles(self) -> int:
        """Get number of unique vehicles in the dataset."""
        vehicle_ids = set()
        for _, vehicle_id, _, _ in self.samples:
            vehicle_ids.add(vehicle_id)
        return len(vehicle_ids)
    
    def get_num_cameras(self) -> int:
        """Get number of unique cameras in the dataset."""
        camera_ids = set()
        for _, _, camera_id, _ in self.samples:
            camera_ids.add(camera_id)
        return len(camera_ids)
    
    def get_vehicle_distribution(self) -> Dict[int, int]:
        """
        Get distribution of images per vehicle.
        
        Returns:
            Dictionary mapping vehicle_id to number of images
        """
        vehicle_counts = {}
        for _, vehicle_id, _, _ in self.samples:
            vehicle_counts[vehicle_id] = vehicle_counts.get(vehicle_id, 0) + 1
        return vehicle_counts
    
    def get_camera_distribution(self) -> Dict[int, int]:
        """
        Get distribution of images per camera.
        
        Returns:
            Dictionary mapping camera_id to number of images
        """
        camera_counts = {}
        for _, _, camera_id, _ in self.samples:
            camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
        return camera_counts
    
    def get_query_gallery_info(self) -> Dict[str, Any]:
        """
        Get information about query-gallery pairs (for evaluation).
        Only applicable for query split.
        
        Returns:
            Dictionary with query-gallery matching information
        """
        if self.split != 'query':
            raise ValueError("Query-gallery info only available for query split")
        
        # Load ground truth file
        gt_file = os.path.join(self.veri_root, 'gt_image.txt')
        gt_info = {}
        
        if os.path.exists(gt_file):
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        query_name = parts[0]
                        gallery_matches = parts[1:]
                        gt_info[query_name] = gallery_matches
        
        return gt_info