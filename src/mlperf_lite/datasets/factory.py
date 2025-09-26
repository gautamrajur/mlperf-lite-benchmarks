"""Dataset factory for MLPerf Lite Benchmarks."""

from typing import Any, Dict, Type
from mlperf_lite.datasets.base import BaseDataset
from mlperf_lite.datasets.cifar10 import CIFAR10Dataset
from mlperf_lite.datasets.sst2 import SST2Dataset


class DatasetFactory:
    """Factory class for creating dataset instances."""
    
    _registry: Dict[str, Type[BaseDataset]] = {
        "cifar10": CIFAR10Dataset,
        "sst2": SST2Dataset,
    }
    
    @classmethod
    def create_dataset(
        cls, 
        dataset_name: str, 
        **kwargs: Any
    ) -> BaseDataset:
        """Create a dataset instance.
        
        Args:
            dataset_name: Name of the dataset to create
            **kwargs: Additional parameters for dataset initialization
            
        Returns:
            Dataset instance
            
        Raises:
            ValueError: If dataset name is not supported
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Available datasets: {available}"
            )
        
        dataset_class = cls._registry[dataset_name]
        return dataset_class(**kwargs)
    
    @classmethod
    def get_supported_datasets(cls) -> list[str]:
        """Get list of supported dataset names.
        
        Returns:
            List of supported dataset names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def register_dataset(
        cls, 
        name: str, 
        dataset_class: Type[BaseDataset]
    ) -> None:
        """Register a new dataset class.
        
        Args:
            name: Name to register the dataset under
            dataset_class: Dataset class to register
        """
        cls._registry[name.lower()] = dataset_class
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Available datasets: {available}"
            )
        
        dataset_class = cls._registry[dataset_name]
        
        # Create a temporary instance to get info (with minimal params)
        try:
            temp_instance = dataset_class(
                data_dir="/tmp", 
                batch_size=1,
                download=False
            )
            return temp_instance.get_dataset_info()
        except Exception:
            # Fallback to basic info if instantiation fails
            return {
                "name": dataset_class.__name__,
                "class": dataset_class.__name__,
                "module": dataset_class.__module__,
            }
