"""Command-line interface for MLPerf Lite Benchmarks."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mlperf_lite.benchmarks.runner import BenchmarkRunner
from mlperf_lite.utils.config import ConfigManager
from mlperf_lite.utils.logging import BenchmarkLogger
from mlperf_lite.utils.system_info import SystemInfo


console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """MLPerf Lite Benchmarks - Production-grade benchmarking framework."""
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--output-dir", "-o", default="results", help="Output directory")
@click.option("--device", "-d", default="cuda", help="Device to use (cuda/cpu)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(config: str, output_dir: str, device: str, verbose: bool) -> None:
    """Run a benchmark."""
    try:
        # Load configuration
        config_manager = ConfigManager()
        if config:
            benchmark_config = config_manager.load_config(config)
        else:
            # Use default config
            benchmark_config = config_manager.get_default_config("resnet_cifar10")
            console.print("[yellow]No config specified, using default ResNet CIFAR-10 config[/yellow]")
        
        # Validate configuration
        errors = config_manager.validate_config(benchmark_config)
        if errors:
            console.print("[red]Configuration validation errors:[/red]")
            for error in errors:
                console.print(f"  - {error}")
            sys.exit(1)
        
        # Setup logging
        logger = BenchmarkLogger(
            name="benchmark",
            log_dir=Path(output_dir) / "logs",
            level="DEBUG" if verbose else "INFO"
        )
        
        # Log system info
        system_info = SystemInfo()
        logger.log_system_info(system_info.get_system_info())
        
        # Log benchmark start
        logger.log_benchmark_start(benchmark_config)
        
        # Run benchmark
        runner = BenchmarkRunner(
            config=benchmark_config,
            output_dir=output_dir,
            device=device
        )
        
        results = runner.run_benchmark()
        
        # Log benchmark end
        logger.log_benchmark_end(results)
        
        # Display summary
        display_results_summary(results)
        
        console.print(f"[green]Benchmark completed successfully![/green]")
        console.print(f"Results saved to: {output_dir}")
        console.print(f"Log file: {logger.get_log_file()}")
        
    except Exception as e:
        console.print(f"[red]Error running benchmark: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--output-dir", "-o", default="results", help="Output directory")
@click.option("--device", "-d", default="cuda", help="Device to use (cuda/cpu)")
@click.option("--num-batches", "-n", default=100, help="Number of batches to run")
@click.option("--warmup-batches", "-w", default=10, help="Number of warmup batches")
def inference(config: str, output_dir: str, device: str, num_batches: int, warmup_batches: int) -> None:
    """Run inference-only benchmark."""
    try:
        # Load configuration
        config_manager = ConfigManager()
        if config:
            benchmark_config = config_manager.load_config(config)
        else:
            benchmark_config = config_manager.get_default_config("resnet_cifar10")
            console.print("[yellow]No config specified, using default ResNet CIFAR-10 config[/yellow]")
        
        # Run inference benchmark
        runner = BenchmarkRunner(
            config=benchmark_config,
            output_dir=output_dir,
            device=device
        )
        
        results = runner.run_inference_benchmark(
            num_batches=num_batches,
            warmup_batches=warmup_batches
        )
        
        # Display results
        display_inference_results(results)
        
        console.print(f"[green]Inference benchmark completed successfully![/green]")
        console.print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"[red]Error running inference benchmark: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("config_name")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def create_config(config_name: str, output: Optional[str]) -> None:
    """Create a configuration file from template."""
    try:
        config_manager = ConfigManager()
        
        if output:
            output_path = Path(output)
        else:
            output_path = Path("configs") / f"{config_name}.yaml"
        
        created_path = config_manager.create_config_file(config_name, output_path)
        
        console.print(f"[green]Configuration file created: {created_path}[/green]")
        
        # Display config info
        config_info = config_manager.get_config_info(created_path)
        display_config_info(config_info)
        
    except Exception as e:
        console.print(f"[red]Error creating configuration: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--config-dir", "-d", default="configs", help="Configuration directory")
def list_configs(config_dir: str) -> None:
    """List available configuration files."""
    try:
        config_manager = ConfigManager(config_dir)
        config_files = config_manager.list_config_files()
        
        if not config_files:
            console.print("[yellow]No configuration files found.[/yellow]")
            return
        
        table = Table(title="Available Configuration Files")
        table.add_column("File", style="cyan")
        table.add_column("Dataset", style="green")
        table.add_column("Model", style="blue")
        table.add_column("Epochs", style="magenta")
        table.add_column("Batch Size", style="yellow")
        table.add_column("Valid", style="red")
        
        for config_file in config_files:
            try:
                config_info = config_manager.get_config_info(config_file)
                table.add_row(
                    config_file.name,
                    str(config_info["dataset"]),
                    str(config_info["model"]),
                    str(config_info["epochs"]),
                    str(config_info["batch_size"]),
                    "✓" if config_info["valid"] else "✗"
                )
            except Exception as e:
                table.add_row(
                    config_file.name,
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "✗"
                )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing configurations: {e}[/red]")
        sys.exit(1)


@main.command()
def system_info() -> None:
    """Display system information."""
    try:
        system_info = SystemInfo()
        info = system_info.get_system_info()
        summary = system_info.get_summary()
        
        # Display summary
        console.print(Panel.fit(
            f"[bold]Platform:[/bold] {summary['platform']}\n"
            f"[bold]Python:[/bold] {summary['python_version']}\n"
            f"[bold]CPU Count:[/bold] {summary['cpu_count']}\n"
            f"[bold]Memory:[/bold] {summary['memory_total_gb']:.1f} GB\n"
            f"[bold]GPU Count:[/bold] {summary['gpu_count']}\n"
            f"[bold]CUDA Available:[/bold] {summary['cuda_available']}",
            title="System Summary"
        ))
        
        # Display GPU information
        gpu_info = system_info.get_gpu_info()
        if gpu_info:
            table = Table(title="GPU Information")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Memory", style="blue")
            table.add_column("Compute Capability", style="magenta")
            
            for gpu in gpu_info:
                table.add_row(
                    str(gpu["id"]),
                    gpu["name"],
                    f"{gpu['memory_total_gb']:.1f} GB",
                    gpu.get("compute_capability", "Unknown")
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting system information: {e}[/red]")
        sys.exit(1)


def display_results_summary(results: Dict[str, Any]) -> None:
    """Display benchmark results summary."""
    console.print("\n[bold]Benchmark Results Summary[/bold]")
    
    # Training results
    if "training_results" in results:
        training_results = results["training_results"]
        console.print(f"Best Validation Accuracy: {training_results.get('best_val_accuracy', 0.0):.4f}")
        console.print(f"Training Time: {training_results.get('training_time', 0.0):.2f} seconds")
        console.print(f"Epochs Completed: {training_results.get('epochs_completed', 0)}")
    
    # Test results
    if "test_results" in results:
        test_results = results["test_results"]
        console.print(f"Test Accuracy: {test_results.get('test_accuracy', 0.0):.4f}")
        console.print(f"Test Loss: {test_results.get('test_loss', 0.0):.4f}")
    
    # Performance metrics
    if "metrics" in results:
        metrics = results["metrics"]
        throughput = metrics.get("throughput", {})
        energy = metrics.get("energy", {})
        
        if "samples_per_sec" in throughput:
            console.print(f"Throughput: {throughput['samples_per_sec']:.2f} samples/sec")
        
        if "total_energy_joules" in energy:
            console.print(f"Total Energy: {energy['total_energy_joules']:.2f} J")
            console.print(f"Average Power: {energy.get('avg_power_watts', 0.0):.2f} W")


def display_inference_results(results: Dict[str, Any]) -> None:
    """Display inference benchmark results."""
    console.print("\n[bold]Inference Benchmark Results[/bold]")
    
    if "inference_metrics" in results:
        metrics = results["inference_metrics"]
        console.print(f"Average Inference Time: {metrics.get('avg_inference_time_ms', 0.0):.2f} ms")
        console.print(f"Throughput: {metrics.get('throughput_samples_per_sec', 0.0):.2f} samples/sec")
        console.print(f"Batch Size: {metrics.get('batch_size', 0)}")
        console.print(f"Batches Run: {metrics.get('num_batches', 0)}")


def display_config_info(config_info: Dict[str, Any]) -> None:
    """Display configuration information."""
    console.print("\n[bold]Configuration Information[/bold]")
    console.print(f"File: {config_info['file_path']}")
    console.print(f"Dataset: {config_info['dataset']}")
    console.print(f"Model: {config_info['model']}")
    console.print(f"Epochs: {config_info['epochs']}")
    console.print(f"Batch Size: {config_info['batch_size']}")
    console.print(f"Valid: {'✓' if config_info['valid'] else '✗'}")
    
    if not config_info['valid'] and config_info['validation_errors']:
        console.print("[red]Validation Errors:[/red]")
        for error in config_info['validation_errors']:
            console.print(f"  - {error}")


if __name__ == "__main__":
    main()
