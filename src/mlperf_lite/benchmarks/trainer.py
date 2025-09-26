"""Training utilities for MLPerf Lite Benchmarks."""

import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from mlperf_lite.models.base import BaseModel
from mlperf_lite.benchmarks.metrics import MetricsCollector


class Trainer:
    """Training utility class for MLPerf Lite Benchmarks."""
    
    def __init__(
        self,
        model: BaseModel,
        metrics_collector: MetricsCollector,
        device: str = "cuda",
        mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        **kwargs: Any
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            metrics_collector: Metrics collector instance
            device: Device to train on
            mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            **kwargs: Additional parameters
        """
        self.model = model
        self.metrics_collector = metrics_collector
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Setup mixed precision
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Move model to device
        self.model.to_device(self.device)
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train_mode()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = 0
        
        # Start epoch timer
        self.metrics_collector.start_timer("epoch")
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}", 
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Start batch timer
            batch_start_time = time.time()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(batch)
                    loss = self.model.compute_loss(outputs, batch["labels"])
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.model.optimizer)
                    self.scaler.update()
                    self.model.optimizer.zero_grad()
            else:
                outputs = self._forward_pass(batch)
                loss = self.model.compute_loss(outputs, batch["labels"])
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.model.optimizer.step()
                    self.model.optimizer.zero_grad()
            
            # Update metrics
            batch_time = time.time() - batch_start_time
            self.metrics_collector.record_batch_time(batch_time)
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            correct_predictions += self._count_correct_predictions(outputs, batch["labels"])
            total_samples += batch["labels"].size(0)
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct_predictions / total_samples:.4f}"
            })
        
        # End epoch timer
        epoch_time = self.metrics_collector.end_timer("epoch")
        self.metrics_collector.record_epoch_time(epoch_time)
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate(
        self, 
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval_mode()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pass(batch)
                        loss = self.model.compute_loss(outputs, batch["labels"])
                else:
                    outputs = self._forward_pass(batch)
                    loss = self.model.compute_loss(outputs, batch["labels"])
                
                total_loss += loss.item()
                correct_predictions += self._count_correct_predictions(outputs, batch["labels"])
                total_samples += batch["labels"].size(0)
                num_batches += 1
        
        # Compute validation metrics
        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def test(
        self, 
        test_loader: DataLoader
    ) -> Tuple[float, float]:
        """Test the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        return self.validate(test_loader)  # Same as validation
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform forward pass based on model type.
        
        Args:
            batch: Input batch
            
        Returns:
            Model outputs
        """
        if "input_ids" in batch and "attention_mask" in batch:
            # BERT model
            return self.model.forward(
                batch["input_ids"], 
                batch["attention_mask"]
            )
        else:
            # ResNet/UNet model
            return self.model.forward(batch["images"])
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch moved to device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _count_correct_predictions(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> int:
        """Count correct predictions.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Number of correct predictions
        """
        if outputs.dim() > 1:
            # Classification task
            predictions = torch.argmax(outputs, dim=1)
            return (predictions == targets).sum().item()
        else:
            # Regression task (if needed in future)
            return 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        target_accuracy: Optional[float] = None,
        early_stopping_patience: int = 10,
        save_checkpoint: bool = True,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, Any]:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            target_accuracy: Target accuracy for early stopping
            early_stopping_patience: Patience for early stopping
            save_checkpoint: Whether to save checkpoints
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training results
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        # Start energy monitoring
        self.metrics_collector.start_energy_monitoring()
        
        # Start training timer
        self.metrics_collector.start_timer("training")
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update metrics
            self.metrics_collector.update_accuracy(epoch, train_acc, val_acc)
            self.metrics_collector.update_loss(train_loss, val_loss)
            self.metrics_collector.metrics.epochs_completed = epoch + 1
            
            # Learning rate scheduling
            if self.model.scheduler:
                self.model.scheduler.step()
            
            # Check for best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                if save_checkpoint:
                    checkpoint_path = f"{checkpoint_dir}/best_model.pth"
                    self.model.save_checkpoint(
                        checkpoint_path, 
                        epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc
                    )
            else:
                patience_counter += 1
            
            # Early stopping
            if target_accuracy and val_acc >= target_accuracy:
                print(f"Target accuracy {target_accuracy} reached at epoch {epoch}")
                break
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # End training timer
        training_time = self.metrics_collector.end_timer("training")
        
        # Stop energy monitoring
        self.metrics_collector.stop_energy_monitoring()
        
        # Finalize metrics
        final_metrics = self.metrics_collector.finalize_metrics()
        
        return {
            "best_val_accuracy": best_val_acc,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "training_time": training_time,
            "epochs_completed": epoch + 1,
            "metrics": final_metrics
        }
