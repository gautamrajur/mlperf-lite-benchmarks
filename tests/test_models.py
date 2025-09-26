"""Tests for model wrappers."""

import pytest
import torch

from mlperf_lite.models.factory import ModelFactory
from mlperf_lite.models.resnet import ResNetModel
from mlperf_lite.models.bert import BERTModel
from mlperf_lite.models.unet import UNetModel


class TestModelFactory:
    """Test model factory functionality."""
    
    def test_get_supported_models(self):
        """Test getting supported models."""
        models = ModelFactory.get_supported_models()
        assert "resnet" in models
        assert "bert" in models
        assert "unet" in models
    
    def test_create_resnet_model(self):
        """Test creating ResNet model."""
        model = ModelFactory.create_model(
            "resnet",
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        assert isinstance(model, ResNetModel)
        assert model.num_classes == 10
        assert model.input_shape == (3, 32, 32)
    
    def test_create_bert_model(self):
        """Test creating BERT model."""
        model = ModelFactory.create_model(
            "bert",
            num_classes=2,
            input_shape=(128,),
            device="cpu"
        )
        assert isinstance(model, BERTModel)
        assert model.num_classes == 2
    
    def test_create_unet_model(self):
        """Test creating UNet model."""
        model = ModelFactory.create_model(
            "unet",
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        assert isinstance(model, UNetModel)
        assert model.num_classes == 10
    
    def test_unsupported_model(self):
        """Test creating unsupported model."""
        with pytest.raises(ValueError, match="Unsupported model"):
            ModelFactory.create_model("unsupported_model")


class TestResNetModel:
    """Test ResNet model functionality."""
    
    def test_model_creation(self):
        """Test ResNet model creation."""
        model = ResNetModel(
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        
        assert model.num_classes == 10
        assert model.input_shape == (3, 32, 32)
        assert model.model is not None
        assert model.optimizer is not None
        assert model.criterion is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = ResNetModel(
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        
        # Create dummy input
        x = torch.randn(2, 3, 32, 32)
        output = model.forward(x)
        
        assert output.shape == (2, 10)
    
    def test_model_info(self):
        """Test model info."""
        model = ResNetModel(
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        
        info = model.get_model_info()
        assert info["num_classes"] == 10
        assert info["input_shape"] == (3, 32, 32)
        assert info["device"] == "cpu"
        assert info["total_parameters"] > 0


class TestBERTModel:
    """Test BERT model functionality."""
    
    def test_model_creation(self):
        """Test BERT model creation."""
        model = BERTModel(
            num_classes=2,
            input_shape=(128,),
            device="cpu"
        )
        
        assert model.num_classes == 2
        assert model.input_shape == (128,)
        assert model.model is not None
        assert model.optimizer is not None
        assert model.criterion is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = BERTModel(
            num_classes=2,
            input_shape=(128,),
            device="cpu"
        )
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        output = model.forward(input_ids, attention_mask)
        
        assert output.shape == (2, 2)
    
    def test_model_info(self):
        """Test model info."""
        model = BERTModel(
            num_classes=2,
            input_shape=(128,),
            device="cpu"
        )
        
        info = model.get_model_info()
        assert info["num_classes"] == 2
        assert info["input_shape"] == (128,)
        assert info["device"] == "cpu"
        assert info["total_parameters"] > 0


class TestUNetModel:
    """Test UNet model functionality."""
    
    def test_model_creation(self):
        """Test UNet model creation."""
        model = UNetModel(
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        
        assert model.num_classes == 10
        assert model.input_shape == (3, 32, 32)
        assert model.model is not None
        assert model.optimizer is not None
        assert model.criterion is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = UNetModel(
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        
        # Create dummy input
        x = torch.randn(2, 3, 32, 32)
        output = model.forward(x)
        
        assert output.shape == (2, 10, 32, 32)
    
    def test_model_info(self):
        """Test model info."""
        model = UNetModel(
            num_classes=10,
            input_shape=(3, 32, 32),
            device="cpu"
        )
        
        info = model.get_model_info()
        assert info["num_classes"] == 10
        assert info["input_shape"] == (3, 32, 32)
        assert info["device"] == "cpu"
        assert info["total_parameters"] > 0
