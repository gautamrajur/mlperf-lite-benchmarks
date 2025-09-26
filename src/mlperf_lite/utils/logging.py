"""Logging utilities for MLPerf Lite Benchmarks."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import time


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        log_format: Custom log format
        include_timestamp: Whether to include timestamp in log format
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Default log format
    if log_format is None:
        if include_timestamp:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            log_format = "%(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


class BenchmarkLogger:
    """Specialized logger for benchmark runs."""
    
    def __init__(
        self,
        name: str = "mlperf_lite",
        log_dir: Union[str, Path] = "logs",
        level: Union[str, int] = logging.INFO
    ) -> None:
        """Initialize benchmark logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def log_benchmark_start(self, config: dict) -> None:
        """Log benchmark start."""
        self.info("=" * 60)
        self.info("MLPerf Lite Benchmark Started")
        self.info("=" * 60)
        self.info(f"Dataset: {config.get('dataset', {}).get('name', 'unknown')}")
        self.info(f"Model: {config.get('model', {}).get('name', 'unknown')}")
        self.info(f"Epochs: {config.get('training', {}).get('epochs', 'unknown')}")
        self.info(f"Batch Size: {config.get('dataset', {}).get('params', {}).get('batch_size', 'unknown')}")
        self.info(f"Device: {config.get('benchmark', {}).get('device', 'unknown')}")
        self.info("=" * 60)
    
    def log_benchmark_end(self, results: dict) -> None:
        """Log benchmark end."""
        self.info("=" * 60)
        self.info("MLPerf Lite Benchmark Completed")
        self.info("=" * 60)
        
        if "training_results" in results:
            training_results = results["training_results"]
            self.info(f"Best Validation Accuracy: {training_results.get('best_val_accuracy', 0.0):.4f}")
            self.info(f"Training Time: {training_results.get('training_time', 0.0):.2f} seconds")
            self.info(f"Epochs Completed: {training_results.get('epochs_completed', 0)}")
        
        if "test_results" in results:
            test_results = results["test_results"]
            self.info(f"Test Accuracy: {test_results.get('test_accuracy', 0.0):.4f}")
            self.info(f"Test Loss: {test_results.get('test_loss', 0.0):.4f}")
        
        if "metrics" in results:
            metrics = results["metrics"]
            throughput = metrics.get("throughput", {})
            energy = metrics.get("energy", {})
            
            if "samples_per_sec" in throughput:
                self.info(f"Throughput: {throughput['samples_per_sec']:.2f} samples/sec")
            
            if "total_energy_joules" in energy:
                self.info(f"Total Energy: {energy['total_energy_joules']:.2f} J")
                self.info(f"Average Power: {energy.get('avg_power_watts', 0.0):.2f} W")
        
        self.info("=" * 60)
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> None:
        """Log epoch results."""
        self.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
    
    def log_system_info(self, system_info: dict) -> None:
        """Log system information."""
        self.info("System Information:")
        self.info(f"  Platform: {system_info.get('platform', {}).get('system', 'unknown')}")
        self.info(f"  Python: {system_info.get('platform', {}).get('python_version', 'unknown')}")
        self.info(f"  CPU Count: {system_info.get('cpu', {}).get('count', 'unknown')}")
        self.info(f"  Memory: {system_info.get('memory', {}).get('total_gb', 'unknown'):.1f} GB")
        
        if "pytorch" in system_info:
            pytorch_info = system_info["pytorch"]
            self.info(f"  PyTorch: {pytorch_info.get('version', 'unknown')}")
            self.info(f"  CUDA Available: {pytorch_info.get('cuda_available', False)}")
            if pytorch_info.get('cuda_available'):
                self.info(f"  CUDA Version: {pytorch_info.get('cuda_version', 'unknown')}")
                self.info(f"  GPU Count: {pytorch_info.get('device_count', 0)}")
    
    def get_log_file(self) -> Path:
        """Get log file path."""
        return self.log_file
