"""CIFAR-10 dataset loader for MLPerf Lite Benchmarks."""

import os
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mlperf_lite.datasets.base import BaseDataset


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset implementation."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
        normalize: bool = True,
        augment: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize CIFAR-10 dataset.
        
        Args:
            data_dir: Directory to store/load CIFAR-10 data
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            download: Whether to download the dataset if not present
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation for training
            **kwargs: Additional parameters
        """
        self.download = download
        self.normalize = normalize
        self.augment = augment
        super().__init__(data_dir, batch_size, num_workers, pin_memory, **kwargs)
    
    def _get_transforms(self, is_training: bool = False) -> transforms.Compose:
        """Get data transforms for CIFAR-10.
        
        Args:
            is_training: Whether transforms are for training (with augmentation)
            
        Returns:
            Composed transforms
        """
        transform_list = []
        
        if is_training and self.augment:
            # Training transforms with augmentation
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if requested
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _load_datasets(self) -> None:
        """Load CIFAR-10 train, validation, and test datasets."""
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Get transforms
        train_transform = self._get_transforms(is_training=True)
        test_transform = self._get_transforms(is_training=False)
        
        # Load datasets
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=train_transform
        )
        
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=test_transform
        )
        
        # Split training set into train/val (80/20 split)
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
        
        # Apply validation transforms to validation set
        self.val_dataset.dataset.transform = test_transform
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_num_classes(self) -> int:
        """Get number of classes in CIFAR-10."""
        return 10
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get input shape for CIFAR-10."""
        return (3, 32, 32)
    
    def get_class_names(self) -> list[str]:
        """Get class names for CIFAR-10."""
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get CIFAR-10 specific dataset information."""
        info = super().get_dataset_info()
        info.update({
            "dataset_type": "CIFAR-10",
            "image_size": "32x32",
            "channels": 3,
            "download": self.download,
            "normalize": self.normalize,
            "augment": self.augment,
        })
        return info
