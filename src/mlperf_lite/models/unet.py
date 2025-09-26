"""UNet model wrapper for MLPerf Lite Benchmarks."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from mlperf_lite.models.base import BaseModel


class DoubleConv(nn.Module):
    """Double convolution block for UNet."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block for UNet."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize downscaling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block for UNet."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize upscaling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.
        
        Args:
            x1: Upsampled feature map
            x2: Skip connection feature map
            
        Returns:
            Concatenated and convolved feature map
        """
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution block for UNet."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize output convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)


class UNetModel(BaseModel):
    """UNet model implementation for semantic segmentation."""
    
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, ...],
        device: str = "cuda",
        in_channels: int = 3,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler: str = "step",
        **kwargs: Any
    ) -> None:
        """Initialize UNet model.
        
        Args:
            num_classes: Number of output classes (including background)
            input_shape: Input tensor shape (C, H, W)
            device: Device to run the model on
            in_channels: Number of input channels
            optimizer: Optimizer type (adam, adamw)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler: Learning rate scheduler (step, cosine)
            **kwargs: Additional parameters
        """
        self.in_channels = in_channels
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        
        super().__init__(num_classes, input_shape, device, **kwargs)
    
    def _build_model(self) -> None:
        """Build UNet model."""
        self.model = UNet(
            n_channels=self.in_channels,
            n_classes=self.num_classes
        )
        
        # Move to device
        self.model = self.model.to(self.device)
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.optimizer_name.lower() == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adamw":
            self.optimizer = AdamW(
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
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        return self.model(x)
    
    def compute_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss for segmentation.
        
        Args:
            outputs: Model outputs of shape (batch_size, num_classes, H, W)
            targets: Ground truth masks of shape (batch_size, H, W)
            
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
        """Get UNet specific model information."""
        info = super().get_model_info()
        info.update({
            "in_channels": self.in_channels,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler_name,
            "model_type": "semantic_segmentation",
        })
        return info


class UNet(nn.Module):
    """UNet architecture implementation."""
    
    def __init__(self, n_channels: int, n_classes: int) -> None:
        """Initialize UNet.
        
        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, n_classes, height, width)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
