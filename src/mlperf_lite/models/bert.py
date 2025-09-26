"""BERT model wrapper for MLPerf Lite Benchmarks."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import AutoModel, AutoConfig

from mlperf_lite.models.base import BaseModel


class BERTModel(BaseModel):
    """BERT model implementation for text classification."""
    
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, ...],
        device: str = "cuda",
        model_name: str = "bert-base-uncased",
        dropout_rate: float = 0.1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        scheduler: str = "linear",
        **kwargs: Any
    ) -> None:
        """Initialize BERT model.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input tensor shape (sequence_length,)
            device: Device to run the model on
            model_name: HuggingFace model name
            dropout_rate: Dropout rate for classification head
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler: Learning rate scheduler (linear, cosine)
            **kwargs: Additional parameters
        """
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        
        super().__init__(num_classes, input_shape, device, **kwargs)
    
    def _build_model(self) -> None:
        """Build BERT model."""
        # Load BERT configuration
        config = AutoConfig.from_pretrained(self.model_name)
        
        # Load BERT model
        self.bert = AutoModel.from_pretrained(self.model_name, config=config)
        
        # Create classification head
        hidden_size = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size // 2, self.num_classes)
        )
        
        # Combine BERT and classifier
        self.model = nn.ModuleDict({
            'bert': self.bert,
            'classifier': self.classifier
        })
        
        # Move to device
        self.model = self.model.to(self.device)
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        # Use AdamW optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _setup_criterion(self) -> None:
        """Setup loss criterion."""
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Get BERT outputs
        bert_outputs = self.model['bert'](
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation for classification
        pooled_output = bert_outputs.pooler_output
        
        # Apply classification head
        logits = self.model['classifier'](pooled_output)
        
        return logits
    
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
    
    def setup_scheduler(self, total_epochs: int, warmup_steps: int = 0) -> None:
        """Setup learning rate scheduler.
        
        Args:
            total_epochs: Total number of training epochs
            warmup_steps: Number of warmup steps for linear scheduler
        """
        if self.scheduler_name.lower() == "linear":
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=total_epochs
            )
        elif self.scheduler_name.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get BERT specific model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler_name,
            "model_type": "text_classification",
            "vocab_size": self.bert.config.vocab_size,
            "hidden_size": self.bert.config.hidden_size,
            "num_layers": self.bert.config.num_hidden_layers,
        })
        return info
