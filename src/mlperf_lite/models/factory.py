"""Model factory for MLPerf Lite Benchmarks."""

from typing import Any, Dict, Type
from mlperf_lite.models.base import BaseModel
from mlperf_lite.models.resnet import ResNetModel
from mlperf_lite.models.bert import BERTModel
from mlperf_lite.models.unet import UNetModel


class ModelFactory:
    """Factory class for creating model instances."""
    
    _registry: Dict[str, Type[BaseModel]] = {
        "resnet": ResNetModel,
        "bert": BERTModel,
        "unet": UNetModel,
    }
    
    @classmethod
    def create_model(
        cls, 
        model_name: str, 
        **kwargs: Any
    ) -> BaseModel:
        """Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not supported
        """
        model_name = model_name.lower()
        
        if model_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {available}"
            )
        
        model_class = cls._registry[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def register_model(
        cls, 
        name: str, 
        model_class: Type[BaseModel]
    ) -> None:
        """Register a new model class.
        
        Args:
            name: Name to register the model under
            model_class: Model class to register
        """
        cls._registry[name.lower()] = model_class
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        model_name = model_name.lower()
        
        if model_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {available}"
            )
        
        model_class = cls._registry[model_name]
        
        # Return basic info without instantiation
        return {
            "name": model_class.__name__,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "description": model_class.__doc__ or "No description available",
        }
    
    @classmethod
    def get_model_architectures(cls, model_name: str) -> list[str]:
        """Get supported architectures for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of supported architectures
        """
        model_name = model_name.lower()
        
        if model_name == "resnet":
            return ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        elif model_name == "bert":
            return [
                "bert-base-uncased",
                "bert-large-uncased", 
                "bert-base-cased",
                "bert-large-cased"
            ]
        elif model_name == "unet":
            return ["unet"]  # UNet doesn't have multiple architectures
        else:
            return []
