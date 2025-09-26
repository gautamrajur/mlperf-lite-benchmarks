"""Tests for configuration management."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path

from mlperf_lite.utils.config import ConfigManager


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def test_config_manager_creation(self):
        """Test config manager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            assert config_manager.config_dir == Path(temp_dir)
    
    def test_get_default_config(self):
        """Test getting default configurations."""
        config_manager = ConfigManager()
        
        # Test ResNet CIFAR-10 config
        config = config_manager.get_default_config("resnet_cifar10")
        assert config["dataset"]["name"] == "cifar10"
        assert config["model"]["name"] == "resnet"
        assert config["training"]["epochs"] == 100
        
        # Test BERT SST-2 config
        config = config_manager.get_default_config("bert_sst2")
        assert config["dataset"]["name"] == "sst2"
        assert config["model"]["name"] == "bert"
        assert config["training"]["epochs"] == 3
    
    def test_unsupported_default_config(self):
        """Test getting unsupported default config."""
        config_manager = ConfigManager()
        
        with pytest.raises(ValueError, match="Unknown default config"):
            config_manager.get_default_config("unsupported")
    
    def test_save_and_load_yaml_config(self):
        """Test saving and loading YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Get default config
            config = config_manager.get_default_config("resnet_cifar10")
            
            # Save config
            config_path = Path(temp_dir) / "test_config.yaml"
            config_manager.save_config(config, config_path)
            
            # Load config
            loaded_config = config_manager.load_config(config_path)
            
            assert loaded_config["dataset"]["name"] == config["dataset"]["name"]
            assert loaded_config["model"]["name"] == config["model"]["name"]
            assert loaded_config["training"]["epochs"] == config["training"]["epochs"]
    
    def test_save_and_load_json_config(self):
        """Test saving and loading JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Get default config
            config = config_manager.get_default_config("resnet_cifar10")
            
            # Save config
            config_path = Path(temp_dir) / "test_config.json"
            config_manager.save_config(config, config_path)
            
            # Load config
            loaded_config = config_manager.load_config(config_path)
            
            assert loaded_config["dataset"]["name"] == config["dataset"]["name"]
            assert loaded_config["model"]["name"] == config["model"]["name"]
            assert loaded_config["training"]["epochs"] == config["training"]["epochs"]
    
    def test_create_config_file(self):
        """Test creating configuration file from template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create config file
            config_path = config_manager.create_config_file("resnet_cifar10")
            
            assert config_path.exists()
            assert config_path.suffix == ".yaml"
            
            # Load and verify
            config = config_manager.load_config(config_path)
            assert config["dataset"]["name"] == "cifar10"
            assert config["model"]["name"] == "resnet"
    
    def test_merge_configs(self):
        """Test merging configurations."""
        config_manager = ConfigManager()
        
        base_config = {
            "dataset": {"name": "cifar10", "batch_size": 32},
            "model": {"name": "resnet", "learning_rate": 0.1},
            "training": {"epochs": 100}
        }
        
        override_config = {
            "dataset": {"batch_size": 64},
            "model": {"learning_rate": 0.01},
            "new_section": {"param": "value"}
        }
        
        merged = config_manager.merge_configs(base_config, override_config)
        
        assert merged["dataset"]["name"] == "cifar10"
        assert merged["dataset"]["batch_size"] == 64
        assert merged["model"]["name"] == "resnet"
        assert merged["model"]["learning_rate"] == 0.01
        assert merged["training"]["epochs"] == 100
        assert merged["new_section"]["param"] == "value"
    
    def test_validate_config(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Valid config
        valid_config = config_manager.get_default_config("resnet_cifar10")
        errors = config_manager.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config - missing dataset
        invalid_config = {
            "model": {"name": "resnet"},
            "training": {"epochs": 100}
        }
        errors = config_manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("Missing required section: dataset" in error for error in errors)
        
        # Invalid config - unknown dataset
        invalid_config = {
            "dataset": {"name": "unknown_dataset"},
            "model": {"name": "resnet"},
            "training": {"epochs": 100}
        }
        errors = config_manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("Unknown dataset" in error for error in errors)
        
        # Invalid config - invalid epochs
        invalid_config = {
            "dataset": {"name": "cifar10"},
            "model": {"name": "resnet"},
            "training": {"epochs": -1}
        }
        errors = config_manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("positive integer" in error for error in errors)
    
    def test_get_config_info(self):
        """Test getting configuration information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create a config file
            config_path = config_manager.create_config_file("resnet_cifar10")
            
            # Get config info
            info = config_manager.get_config_info(config_path)
            
            assert info["file_path"] == str(config_path)
            assert info["dataset"] == "cifar10"
            assert info["model"] == "resnet"
            assert info["epochs"] == 100
            assert info["valid"] is True
            assert len(info["validation_errors"]) == 0
