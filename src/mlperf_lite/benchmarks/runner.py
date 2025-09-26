"""Main benchmark runner for MLPerf Lite Benchmarks."""

import os
import json
import time
from typing import Any, Dict, Optional, Tuple
import torch
from pathlib import Path

from mlperf_lite.datasets.factory import DatasetFactory
from mlperf_lite.models.factory import ModelFactory
from mlperf_lite.benchmarks.trainer import Trainer
from mlperf_lite.benchmarks.metrics import MetricsCollector
from mlperf_lite.utils.system_info import SystemInfo


class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "results",
        device: str = "cuda",
        **kwargs: Any
    ) -> None:
        """Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
            output_dir: Directory to save results
            device: Device to run benchmark on
            **kwargs: Additional parameters
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        self.kwargs = kwargs
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dataset = None
        self.model = None
        self.trainer = None
        self.metrics_collector = None
        
        # System information
        self.system_info = SystemInfo()
    
    def setup(self) -> None:
        """Setup benchmark components."""
        print("Setting up benchmark...")
        
        # Setup dataset
        dataset_config = self.config.get("dataset", {})
        self.dataset = DatasetFactory.create_dataset(
            dataset_config.get("name", "cifar10"),
            **dataset_config.get("params", {})
        )
        
        # Setup model
        model_config = self.config.get("model", {})
        self.model = ModelFactory.create_model(
            model_config.get("name", "resnet"),
            num_classes=self.dataset.get_num_classes(),
            input_shape=self.dataset.get_input_shape(),
            device=self.device,
            **model_config.get("params", {})
        )
        
        # Setup metrics collector
        self.metrics_collector = MetricsCollector(device=self.device)
        
        # Setup trainer
        training_config = self.config.get("training", {})
        self.trainer = Trainer(
            model=self.model,
            metrics_collector=self.metrics_collector,
            device=self.device,
            **training_config.get("params", {})
        )
        
        # Update model info in metrics
        self.metrics_collector.update_model_info(self.model.model)
        
        print("Benchmark setup complete!")
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark.
        
        Returns:
            Benchmark results
        """
        print("Starting benchmark...")
        
        # Setup benchmark
        self.setup()
        
        # Get data loaders
        train_loader = self.dataset.get_train_loader()
        val_loader = self.dataset.get_val_loader()
        test_loader = self.dataset.get_test_loader()
        
        # Training configuration
        training_config = self.config.get("training", {})
        epochs = training_config.get("epochs", 100)
        target_accuracy = training_config.get("target_accuracy")
        early_stopping_patience = training_config.get("early_stopping_patience", 10)
        
        # Setup scheduler if needed
        if hasattr(self.model, 'setup_scheduler'):
            self.model.setup_scheduler(epochs)
        
        # Start benchmark timer
        benchmark_start_time = time.time()
        
        # Train model
        training_results = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            target_accuracy=target_accuracy,
            early_stopping_patience=early_stopping_patience,
            save_checkpoint=True,
            checkpoint_dir=str(self.output_dir / "checkpoints")
        )
        
        # Test model
        test_loss, test_acc = self.trainer.test(test_loader)
        self.metrics_collector.update_test_metrics(test_acc, test_loss)
        
        # Compute throughput
        total_samples = len(self.dataset.train_dataset) + len(self.dataset.val_dataset)
        total_time = time.time() - benchmark_start_time
        self.metrics_collector.compute_throughput(total_samples, total_time)
        
        # Finalize metrics
        final_metrics = self.metrics_collector.finalize_metrics()
        
        # Prepare results
        results = {
            "config": self.config,
            "system_info": self.system_info.get_system_info(),
            "dataset_info": self.dataset.get_dataset_info(),
            "model_info": self.model.get_model_info(),
            "training_results": training_results,
            "test_results": {
                "test_accuracy": test_acc,
                "test_loss": test_loss
            },
            "metrics": final_metrics.to_dict(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save results
        self._save_results(results)
        
        print("Benchmark completed!")
        return results
    
    def run_inference_benchmark(
        self, 
        num_batches: int = 100,
        warmup_batches: int = 10
    ) -> Dict[str, Any]:
        """Run inference-only benchmark.
        
        Args:
            num_batches: Number of batches to run
            warmup_batches: Number of warmup batches
            
        Returns:
            Inference benchmark results
        """
        print("Starting inference benchmark...")
        
        # Setup benchmark
        self.setup()
        
        # Get test loader
        test_loader = self.dataset.get_test_loader()
        
        # Load best model if available
        checkpoint_path = self.output_dir / "checkpoints" / "best_model.pth"
        if checkpoint_path.exists():
            self.model.load_checkpoint(str(checkpoint_path))
        
        self.model.eval_mode()
        
        # Warmup
        print(f"Warming up with {warmup_batches} batches...")
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= warmup_batches:
                    break
                batch = self.trainer._move_batch_to_device(batch)
                _ = self.trainer._forward_pass(batch)
        
        # Benchmark inference
        print(f"Running inference benchmark with {num_batches} batches...")
        inference_times = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                batch = self.trainer._move_batch_to_device(batch)
                
                # Time inference
                start_time = time.time()
                outputs = self.trainer._forward_pass(batch)
                torch.cuda.synchronize() if self.device == "cuda" else None
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
        
        # Compute metrics
        avg_inference_time = sum(inference_times) / len(inference_times)
        throughput = self.dataset.batch_size / avg_inference_time
        
        results = {
            "config": self.config,
            "system_info": self.system_info.get_system_info(),
            "model_info": self.model.get_model_info(),
            "inference_metrics": {
                "avg_inference_time_ms": avg_inference_time * 1000,
                "throughput_samples_per_sec": throughput,
                "batch_size": self.dataset.batch_size,
                "num_batches": num_batches,
                "warmup_batches": warmup_batches,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save results
        self._save_results(results, suffix="_inference")
        
        print("Inference benchmark completed!")
        return results
    
    def _save_results(self, results: Dict[str, Any], suffix: str = "") -> None:
        """Save benchmark results.
        
        Args:
            results: Results to save
            suffix: Suffix for filename
        """
        # Save JSON results
        results_file = self.output_dir / f"results{suffix}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save config
        config_file = self.output_dir / f"config{suffix}.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary.
        
        Returns:
            Summary of benchmark results
        """
        if not self.metrics_collector:
            return {}
        
        return self.metrics_collector.get_summary()
