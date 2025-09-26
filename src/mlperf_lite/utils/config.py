"""Configuration management for MLPerf Lite Benchmarks."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class ConfigManager:
    """Manage configuration files and parameters."""
    
    def __init__(self, config_dir: str = "configs") -> None:
        """Initialize config manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self._default_configs = {
            "resnet_cifar10": self._get_resnet_cifar10_config(),
            "bert_sst2": self._get_bert_sst2_config(),
            "unet_segmentation": self._get_unet_segmentation_config(),
        }
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Get default configuration by name.
        
        Args:
            config_name: Name of the default configuration
            
        Returns:
            Default configuration dictionary
        """
        if config_name not in self._default_configs:
            available = ", ".join(self._default_configs.keys())
            raise ValueError(f"Unknown default config: {config_name}. Available: {available}")
        
        return self._default_configs[config_name].copy()
    
    def create_config_file(self, config_name: str, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Create a configuration file from default template.
        
        Args:
            config_name: Name of the default configuration
            output_path: Output path for the configuration file
            
        Returns:
            Path to created configuration file
        """
        config = self.get_default_config(config_name)
        
        if output_path is None:
            output_path = self.config_dir / f"{config_name}.yaml"
        
        output_path = Path(output_path)
        self.save_config(config, output_path)
        
        return output_path
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations, with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Required sections
        required_sections = ["dataset", "model", "training"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate dataset section
        if "dataset" in config:
            dataset_config = config["dataset"]
            if "name" not in dataset_config:
                errors.append("Dataset section missing 'name' field")
            elif dataset_config["name"] not in ["cifar10", "sst2"]:
                errors.append(f"Unknown dataset: {dataset_config['name']}")
        
        # Validate model section
        if "model" in config:
            model_config = config["model"]
            if "name" not in model_config:
                errors.append("Model section missing 'name' field")
            elif model_config["name"] not in ["resnet", "bert", "unet"]:
                errors.append(f"Unknown model: {model_config['name']}")
        
        # Validate training section
        if "training" in config:
            training_config = config["training"]
            if "epochs" not in training_config:
                errors.append("Training section missing 'epochs' field")
            elif not isinstance(training_config["epochs"], int) or training_config["epochs"] <= 0:
                errors.append("Training epochs must be a positive integer")
        
        return errors
    
    def _get_resnet_cifar10_config(self) -> Dict[str, Any]:
        """Get default ResNet CIFAR-10 configuration."""
        return {
            "dataset": {
                "name": "cifar10",
                "params": {
                    "data_dir": "./data/cifar10",
                    "batch_size": 128,
                    "num_workers": 4,
                    "download": True,
                    "normalize": True,
                    "augment": True,
                }
            },
            "model": {
                "name": "resnet",
                "params": {
                    "architecture": "resnet18",
                    "pretrained": False,
                    "optimizer": "sgd",
                    "learning_rate": 0.1,
                    "weight_decay": 1e-4,
                    "momentum": 0.9,
                    "scheduler": "step",
                }
            },
            "training": {
                "epochs": 100,
                "target_accuracy": 0.95,
                "early_stopping_patience": 10,
                "params": {
                    "mixed_precision": False,
                    "gradient_accumulation_steps": 1,
                }
            },
            "benchmark": {
                "output_dir": "./results/resnet_cifar10",
                "device": "cuda",
                "save_checkpoints": True,
                "monitor_energy": True,
                "monitor_hardware": True,
            }
        }
    
    def _get_bert_sst2_config(self) -> Dict[str, Any]:
        """Get default BERT SST-2 configuration."""
        return {
            "dataset": {
                "name": "sst2",
                "params": {
                    "data_dir": "./data/sst2",
                    "batch_size": 32,
                    "num_workers": 4,
                    "model_name": "bert-base-uncased",
                    "max_length": 128,
                }
            },
            "model": {
                "name": "bert",
                "params": {
                    "model_name": "bert-base-uncased",
                    "dropout_rate": 0.1,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "scheduler": "linear",
                }
            },
            "training": {
                "epochs": 3,
                "target_accuracy": 0.90,
                "early_stopping_patience": 3,
                "params": {
                    "mixed_precision": True,
                    "gradient_accumulation_steps": 1,
                }
            },
            "benchmark": {
                "output_dir": "./results/bert_sst2",
                "device": "cuda",
                "save_checkpoints": True,
                "monitor_energy": True,
                "monitor_hardware": True,
            }
        }
    
    def _get_unet_segmentation_config(self) -> Dict[str, Any]:
        """Get default UNet segmentation configuration."""
        return {
            "dataset": {
                "name": "cifar10",  # Using CIFAR-10 as placeholder, would need custom segmentation dataset
                "params": {
                    "data_dir": "./data/segmentation",
                    "batch_size": 16,
                    "num_workers": 4,
                    "download": True,
                    "normalize": True,
                    "augment": True,
                }
            },
            "model": {
                "name": "unet",
                "params": {
                    "in_channels": 3,
                    "optimizer": "adam",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                    "scheduler": "step",
                }
            },
            "training": {
                "epochs": 50,
                "target_accuracy": 0.85,
                "early_stopping_patience": 10,
                "params": {
                    "mixed_precision": False,
                    "gradient_accumulation_steps": 1,
                }
            },
            "benchmark": {
                "output_dir": "./results/unet_segmentation",
                "device": "cuda",
                "save_checkpoints": True,
                "monitor_energy": True,
                "monitor_hardware": True,
            }
        }
    
    def list_config_files(self) -> List[Path]:
        """List all configuration files in the config directory.
        
        Returns:
            List of configuration file paths
        """
        config_files = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            config_files.extend(self.config_dir.glob(ext))
        return sorted(config_files)
    
    def get_config_info(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with configuration information
        """
        config_path = Path(config_path)
        config = self.load_config(config_path)
        
        info = {
            "file_path": str(config_path),
            "file_size": config_path.stat().st_size,
            "modified_time": config_path.stat().st_mtime,
            "dataset": config.get("dataset", {}).get("name", "unknown"),
            "model": config.get("model", {}).get("name", "unknown"),
            "epochs": config.get("training", {}).get("epochs", "unknown"),
            "batch_size": config.get("dataset", {}).get("params", {}).get("batch_size", "unknown"),
        }
        
        # Add validation status
        errors = self.validate_config(config)
        info["valid"] = len(errors) == 0
        info["validation_errors"] = errors
        
        return info
