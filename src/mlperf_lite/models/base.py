"""Base model class for MLPerf Lite Benchmarks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class BaseModel(ABC):
    """Abstract base class for all models in MLPerf Lite Benchmarks."""
    
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, ...],
        device: str = "cuda",
        **kwargs: Any
    ) -> None:
        """Initialize the model.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input tensor shape
            device: Device to run the model on
            **kwargs: Additional model-specific parameters
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.device = device
        self.kwargs = kwargs
        
        # Initialize model components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
        # Build the model
        self._build_model()
        self._setup_optimizer()
        self._setup_criterion()
    
    @abstractmethod
    def _build_model(self) -> None:
        """Build the neural network model."""
        pass
    
    @abstractmethod
    def _setup_optimizer(self) -> None:
        """Setup the optimizer."""
        pass
    
    @abstractmethod
    def _setup_criterion(self) -> None:
        """Setup the loss criterion."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between outputs and targets.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Loss tensor
        """
        pass
    
    def to_device(self, device: str) -> None:
        """Move model to specified device.
        
        Args:
            device: Target device
        """
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        if self.criterion is not None:
            self.criterion = self.criterion.to(device)
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata
        """
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
        
        return {
            "name": self.__class__.__name__,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "optimizer": self.optimizer.__class__.__name__ if self.optimizer else None,
            "criterion": self.criterion.__class__.__name__ if self.criterion else None,
        }
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs: Any) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            **kwargs: Additional data to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if self.model else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "model_info": self.get_model_info(),
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint
    
    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return (
            f"{self.__class__.__name__}("
            f"num_classes={info['num_classes']}, "
            f"input_shape={info['input_shape']}, "
            f"device='{info['device']}', "
            f"params={info['total_parameters']:,})"
        )
