"""Tests for dataset loaders."""

import pytest
import tempfile
import shutil
from pathlib import Path

from mlperf_lite.datasets.factory import DatasetFactory
from mlperf_lite.datasets.cifar10 import CIFAR10Dataset
from mlperf_lite.datasets.sst2 import SST2Dataset


class TestDatasetFactory:
    """Test dataset factory functionality."""
    
    def test_get_supported_datasets(self):
        """Test getting supported datasets."""
        datasets = DatasetFactory.get_supported_datasets()
        assert "cifar10" in datasets
        assert "sst2" in datasets
    
    def test_create_cifar10_dataset(self):
        """Test creating CIFAR-10 dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = DatasetFactory.create_dataset(
                "cifar10",
                data_dir=temp_dir,
                batch_size=32,
                download=True
            )
            assert isinstance(dataset, CIFAR10Dataset)
            assert dataset.get_num_classes() == 10
            assert dataset.get_input_shape() == (3, 32, 32)
    
    def test_create_sst2_dataset(self):
        """Test creating SST-2 dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = DatasetFactory.create_dataset(
                "sst2",
                data_dir=temp_dir,
                batch_size=16,
                model_name="bert-base-uncased"
            )
            assert isinstance(dataset, SST2Dataset)
            assert dataset.get_num_classes() == 2
    
    def test_unsupported_dataset(self):
        """Test creating unsupported dataset."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            DatasetFactory.create_dataset("unsupported_dataset")


class TestCIFAR10Dataset:
    """Test CIFAR-10 dataset functionality."""
    
    def test_dataset_creation(self):
        """Test CIFAR-10 dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = CIFAR10Dataset(
                data_dir=temp_dir,
                batch_size=32,
                download=True
            )
            
            assert dataset.get_num_classes() == 10
            assert dataset.get_input_shape() == (3, 32, 32)
            assert len(dataset.get_class_names()) == 10
    
    def test_dataset_info(self):
        """Test dataset info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = CIFAR10Dataset(
                data_dir=temp_dir,
                batch_size=32,
                download=True
            )
            
            info = dataset.get_dataset_info()
            assert info["name"] == "CIFAR10Dataset"
            assert info["num_classes"] == 10
            assert info["input_shape"] == (3, 32, 32)
            assert info["dataset_type"] == "CIFAR-10"


class TestSST2Dataset:
    """Test SST-2 dataset functionality."""
    
    def test_dataset_creation(self):
        """Test SST-2 dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = SST2Dataset(
                data_dir=temp_dir,
                batch_size=16,
                model_name="bert-base-uncased"
            )
            
            assert dataset.get_num_classes() == 2
            assert len(dataset.get_class_names()) == 2
    
    def test_dataset_info(self):
        """Test dataset info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = SST2Dataset(
                data_dir=temp_dir,
                batch_size=16,
                model_name="bert-base-uncased"
            )
            
            info = dataset.get_dataset_info()
            assert info["name"] == "SST2Dataset"
            assert info["num_classes"] == 2
            assert info["dataset_type"] == "SST-2"
