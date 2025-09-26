"""Metrics collection for MLPerf Lite Benchmarks."""

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from mlperf_lite.monitoring import EnergyMonitor


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    
    # Timing metrics
    total_time: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    data_loading_time: float = 0.0
    
    # Throughput metrics
    throughput_samples_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0  # For text models
    
    # Accuracy metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    best_accuracy: float = 0.0
    accuracy_at_epoch: Dict[int, float] = field(default_factory=dict)
    
    # Loss metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    test_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    # Energy metrics
    total_energy_joules: float = 0.0
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    
    # Hardware metrics
    gpu_utilization: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)
    
    # Training progress
    epochs_completed: int = 0
    total_epochs: int = 0
    convergence_epoch: Optional[int] = None
    
    # Model metrics
    model_size_mb: float = 0.0
    parameters_count: int = 0
    flops: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timing": {
                "total_time": self.total_time,
                "training_time": self.training_time,
                "inference_time": self.inference_time,
                "data_loading_time": self.data_loading_time,
            },
            "throughput": {
                "samples_per_sec": self.throughput_samples_per_sec,
                "tokens_per_sec": self.throughput_tokens_per_sec,
            },
            "accuracy": {
                "train_accuracy": self.train_accuracy,
                "val_accuracy": self.val_accuracy,
                "test_accuracy": self.test_accuracy,
                "best_accuracy": self.best_accuracy,
                "accuracy_at_epoch": self.accuracy_at_epoch,
            },
            "loss": {
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "test_loss": self.test_loss,
                "loss_history": self.loss_history,
            },
            "energy": {
                "total_energy_joules": self.total_energy_joules,
                "avg_power_watts": self.avg_power_watts,
                "peak_power_watts": self.peak_power_watts,
            },
            "hardware": {
                "gpu_utilization": self.gpu_utilization,
                "memory_usage_mb": self.memory_usage_mb,
                "gpu_temperature": self.gpu_temperature,
            },
            "training": {
                "epochs_completed": self.epochs_completed,
                "total_epochs": self.total_epochs,
                "convergence_epoch": self.convergence_epoch,
            },
            "model": {
                "model_size_mb": self.model_size_mb,
                "parameters_count": self.parameters_count,
                "flops": self.flops,
            }
        }


class MetricsCollector:
    """Collects and tracks benchmark metrics."""
    
    def __init__(self, device: str = "cuda") -> None:
        """Initialize metrics collector.
        
        Args:
            device: Device to monitor
        """
        self.device = device
        self.metrics = BenchmarkMetrics()
        self.energy_monitor = EnergyMonitor(device) if device == "cuda" else None
        
        # Timing state
        self._timers: Dict[str, float] = {}
        self._batch_times: List[float] = []
        self._epoch_times: List[float] = []
        
        # Accuracy tracking
        self._best_accuracy = 0.0
        self._convergence_threshold = 0.95  # 95% of target accuracy
        
    def start_timer(self, name: str) -> None:
        """Start a named timer.
        
        Args:
            name: Timer name
        """
        self._timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self._timers:
            return 0.0
        
        elapsed = time.time() - self._timers[name]
        del self._timers[name]
        return elapsed
    
    def record_batch_time(self, batch_time: float) -> None:
        """Record batch processing time.
        
        Args:
            batch_time: Time taken for one batch
        """
        self._batch_times.append(batch_time)
    
    def record_epoch_time(self, epoch_time: float) -> None:
        """Record epoch processing time.
        
        Args:
            epoch_time: Time taken for one epoch
        """
        self._epoch_times.append(epoch_time)
    
    def update_accuracy(self, epoch: int, train_acc: float, val_acc: float) -> None:
        """Update accuracy metrics.
        
        Args:
            epoch: Current epoch
            train_acc: Training accuracy
            val_acc: Validation accuracy
        """
        self.metrics.train_accuracy = train_acc
        self.metrics.val_accuracy = val_acc
        self.metrics.accuracy_at_epoch[epoch] = val_acc
        
        # Track best accuracy
        if val_acc > self._best_accuracy:
            self._best_accuracy = val_acc
            self.metrics.best_accuracy = self._best_accuracy
    
    def update_loss(self, train_loss: float, val_loss: float) -> None:
        """Update loss metrics.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
        """
        self.metrics.train_loss = train_loss
        self.metrics.val_loss = val_loss
        self.metrics.loss_history.append(val_loss)
    
    def update_test_metrics(self, test_acc: float, test_loss: float) -> None:
        """Update test metrics.
        
        Args:
            test_acc: Test accuracy
            test_loss: Test loss
        """
        self.metrics.test_accuracy = test_acc
        self.metrics.test_loss = test_loss
    
    def update_model_info(self, model: torch.nn.Module) -> None:
        """Update model information.
        
        Args:
            model: PyTorch model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.metrics.parameters_count = total_params
        
        # Estimate model size (rough approximation)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        self.metrics.model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    def compute_throughput(self, num_samples: int, total_time: float) -> None:
        """Compute throughput metrics.
        
        Args:
            num_samples: Number of samples processed
            total_time: Total processing time
        """
        self.metrics.throughput_samples_per_sec = num_samples / total_time if total_time > 0 else 0.0
    
    def compute_throughput_tokens(self, num_tokens: int, total_time: float) -> None:
        """Compute token throughput for text models.
        
        Args:
            num_tokens: Number of tokens processed
            total_time: Total processing time
        """
        self.metrics.throughput_tokens_per_sec = num_tokens / total_time if total_time > 0 else 0.0
    
    def start_energy_monitoring(self) -> None:
        """Start energy monitoring."""
        if self.energy_monitor:
            self.energy_monitor.start_monitoring()
    
    def stop_energy_monitoring(self) -> None:
        """Stop energy monitoring and update metrics."""
        if self.energy_monitor:
            energy_data = self.energy_monitor.stop_monitoring()
            self.metrics.total_energy_joules = energy_data.get("total_energy", 0.0)
            self.metrics.avg_power_watts = energy_data.get("avg_power", 0.0)
            self.metrics.peak_power_watts = energy_data.get("peak_power", 0.0)
    
    def update_hardware_metrics(self, gpu_util: float, memory_mb: float, temp: float) -> None:
        """Update hardware metrics.
        
        Args:
            gpu_util: GPU utilization percentage
            memory_mb: Memory usage in MB
            temp: GPU temperature in Celsius
        """
        self.metrics.gpu_utilization.append(gpu_util)
        self.metrics.memory_usage_mb.append(memory_mb)
        self.metrics.gpu_temperature.append(temp)
    
    def check_convergence(self, target_accuracy: float) -> bool:
        """Check if model has converged to target accuracy.
        
        Args:
            target_accuracy: Target accuracy threshold
            
        Returns:
            True if converged
        """
        if self.metrics.val_accuracy >= target_accuracy * self._convergence_threshold:
            if self.metrics.convergence_epoch is None:
                self.metrics.convergence_epoch = self.metrics.epochs_completed
            return True
        return False
    
    def finalize_metrics(self) -> BenchmarkMetrics:
        """Finalize and return collected metrics.
        
        Returns:
            Finalized metrics object
        """
        # Compute final timing metrics
        if self._batch_times:
            avg_batch_time = np.mean(self._batch_times)
            self.metrics.inference_time = avg_batch_time
        
        if self._epoch_times:
            self.metrics.training_time = sum(self._epoch_times)
            self.metrics.total_time = self.metrics.training_time + self.metrics.inference_time
        
        # Update epochs completed
        self.metrics.epochs_completed = len(self._epoch_times)
        
        return self.metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics.
        
        Returns:
            Dictionary with key metrics
        """
        return {
            "best_accuracy": self.metrics.best_accuracy,
            "throughput_samples_per_sec": self.metrics.throughput_samples_per_sec,
            "total_energy_joules": self.metrics.total_energy_joules,
            "avg_power_watts": self.metrics.avg_power_watts,
            "model_size_mb": self.metrics.model_size_mb,
            "parameters_count": self.metrics.parameters_count,
            "epochs_completed": self.metrics.epochs_completed,
            "convergence_epoch": self.metrics.convergence_epoch,
        }
