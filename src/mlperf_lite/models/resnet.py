"""ResNet model wrapper for MLPerf Lite Benchmarks."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision.models as models

from mlperf_lite.models.base import BaseModel


class ResNetModel(BaseModel):
    """ResNet model implementation for image classification."""
    
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, ...],
        device: str = "cuda",
        architecture: str = "resnet18",
        pretrained: bool = False,
        optimizer: str = "sgd",
        learning_rate: float = 0.1,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        scheduler: str = "step",
        **kwargs: Any
    ) -> None:
        """Initialize ResNet model.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input tensor shape (C, H, W)
            device: Device to run the model on
            architecture: ResNet architecture (resnet18, resnet34, resnet50, etc.)
            pretrained: Whether to use pretrained weights
            optimizer: Optimizer type (sgd, adam)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            momentum: Momentum for SGD
            scheduler: Learning rate scheduler (step, cosine)
            **kwargs: Additional parameters
        """
        self.architecture = architecture
        self.pretrained = pretrained
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.scheduler_name = scheduler
        
        super().__init__(num_classes, input_shape, device, **kwargs)
    
    def _build_model(self) -> None:
        """Build ResNet model."""
        # Get ResNet architecture
        if not hasattr(models, self.architecture):
            raise ValueError(f"Unsupported ResNet architecture: {self.architecture}")
        
        # Create model
        model_fn = getattr(models, self.architecture)
        self.model = model_fn(pretrained=self.pretrained)
        
        # Modify final layer for number of classes
        if hasattr(self.model, 'fc'):
            # Standard ResNet
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.model, 'classifier'):
            # Some variants
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Cannot modify final layer for {self.architecture}")
        
        # Move to device
        self.model = self.model.to(self.device)
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.optimizer_name.lower() == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _setup_criterion(self) -> None:
        """Setup loss criterion."""
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def compute_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss.
        
        Args:
            outputs: Model outputs of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Loss tensor
        """
        return self.criterion(outputs, targets)
    
    def setup_scheduler(self, total_epochs: int) -> None:
        """Setup learning rate scheduler.
        
        Args:
            total_epochs: Total number of training epochs
        """
        if self.scheduler_name.lower() == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=total_epochs // 3,
                gamma=0.1
            )
        elif self.scheduler_name.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ResNet specific model information."""
        info = super().get_model_info()
        info.update({
            "architecture": self.architecture,
            "pretrained": self.pretrained,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "scheduler": self.scheduler_name,
            "model_type": "image_classification",
        })
        return info
