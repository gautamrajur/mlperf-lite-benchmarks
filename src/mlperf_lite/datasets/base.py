"""Base dataset class for MLPerf Lite Benchmarks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset


class BaseDataset(ABC):
    """Abstract base class for all datasets in MLPerf Lite Benchmarks."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            **kwargs: Additional dataset-specific parameters
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        
        # Initialize dataset-specific attributes
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        # Load the datasets
        self._load_datasets()
    
    @abstractmethod
    def _load_datasets(self) -> None:
        """Load train, validation, and test datasets."""
        pass
    
    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        pass
    
    @abstractmethod
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        pass
    
    @abstractmethod
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Get number of classes in the dataset."""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get input shape for the dataset."""
        pass
    
    @abstractmethod
    def get_class_names(self) -> list[str]:
        """Get class names for the dataset."""
        pass
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information.
        
        Returns:
            Dictionary containing dataset metadata
        """
        return {
            "name": self.__class__.__name__,
            "data_dir": self.data_dir,
            "batch_size": self.batch_size,
            "num_classes": self.get_num_classes(),
            "input_shape": self.get_input_shape(),
            "class_names": self.get_class_names(),
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
        }
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        info = self.get_dataset_info()
        return (
            f"{self.__class__.__name__}("
            f"data_dir='{info['data_dir']}', "
            f"batch_size={info['batch_size']}, "
            f"num_classes={info['num_classes']}, "
            f"input_shape={info['input_shape']})"
        )
