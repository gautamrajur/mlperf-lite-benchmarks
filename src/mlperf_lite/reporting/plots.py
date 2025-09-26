"""Plot generation for MLPerf Lite Benchmarks."""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class PlotGenerator:
    """Generate plots for benchmark results."""
    
    def __init__(self, style: str = "default") -> None:
        """Initialize plot generator.
        
        Args:
            style: Plot style (default, seaborn, plotly)
        """
        self.style = style
        
        if style == "seaborn":
            try:
                import seaborn as sns
                sns.set_style("whitegrid")
                plt.style.use("seaborn")
            except ImportError:
                print("Warning: seaborn not available, using default style")
    
    def plot_training_curves(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot training curves (loss and accuracy).
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save plot
            format: Plot format (png, pdf, svg)
        """
        if self.style == "plotly":
            self._plot_training_curves_plotly(metrics, save_path)
        else:
            self._plot_training_curves_matplotlib(metrics, save_path, format)
    
    def _plot_training_curves_matplotlib(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot training curves using matplotlib."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        loss_history = metrics.get("loss", {}).get("loss_history", [])
        if loss_history:
            ax1.plot(loss_history, label="Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training Loss")
            ax1.legend()
            ax1.grid(True)
        
        # Plot accuracy
        accuracy_at_epoch = metrics.get("accuracy", {}).get("accuracy_at_epoch", {})
        if accuracy_at_epoch:
            epochs = list(accuracy_at_epoch.keys())
            accuracies = list(accuracy_at_epoch.values())
            ax2.plot(epochs, accuracies, label="Validation Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Training Accuracy")
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=format, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_training_curves_plotly(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None
    ) -> None:
        """Plot training curves using plotly."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training Loss", "Training Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot loss
        loss_history = metrics.get("loss", {}).get("loss_history", [])
        if loss_history:
            fig.add_trace(
                go.Scatter(
                    y=loss_history,
                    mode="lines",
                    name="Validation Loss",
                    line=dict(color="blue")
                ),
                row=1, col=1
            )
        
        # Plot accuracy
        accuracy_at_epoch = metrics.get("accuracy", {}).get("accuracy_at_epoch", {})
        if accuracy_at_epoch:
            epochs = list(accuracy_at_epoch.keys())
            accuracies = list(accuracy_at_epoch.values())
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=accuracies,
                    mode="lines",
                    name="Validation Accuracy",
                    line=dict(color="green")
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Training Progress",
            showlegend=True,
            height=400
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_throughput_comparison(
        self, 
        results: List[Dict[str, Any]], 
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot throughput comparison across different runs.
        
        Args:
            results: List of benchmark results
            save_path: Path to save plot
            format: Plot format (png, pdf, svg)
        """
        if self.style == "plotly":
            self._plot_throughput_comparison_plotly(results, save_path)
        else:
            self._plot_throughput_comparison_matplotlib(results, save_path, format)
    
    def _plot_throughput_comparison_matplotlib(
        self, 
        results: List[Dict[str, Any]], 
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot throughput comparison using matplotlib."""
        if not results:
            return
        
        # Extract data
        labels = []
        throughputs = []
        
        for result in results:
            config = result.get("config", {})
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            model_name = config.get("model", {}).get("name", "unknown")
            labels.append(f"{model_name}_{dataset_name}")
            
            throughput = result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0)
            throughputs.append(throughput)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, throughputs)
        
        # Add value labels on bars
        for bar, throughput in zip(bars, throughputs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{throughput:.1f}', ha='center', va='bottom')
        
        ax.set_xlabel("Model-Dataset Combination")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Throughput Comparison")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=format, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_throughput_comparison_plotly(
        self, 
        results: List[Dict[str, Any]], 
        save_path: Optional[str] = None
    ) -> None:
        """Plot throughput comparison using plotly."""
        if not results:
            return
        
        # Extract data
        labels = []
        throughputs = []
        
        for result in results:
            config = result.get("config", {})
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            model_name = config.get("model", {}).get("name", "unknown")
            labels.append(f"{model_name}_{dataset_name}")
            
            throughput = result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0)
            throughputs.append(throughput)
        
        fig = go.Figure(data=[
            go.Bar(x=labels, y=throughputs, text=[f"{t:.1f}" for t in throughputs], textposition='auto')
        ])
        
        fig.update_layout(
            title="Throughput Comparison",
            xaxis_title="Model-Dataset Combination",
            yaxis_title="Throughput (samples/sec)",
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_energy_usage(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot energy usage over time.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save plot
            format: Plot format (png, pdf, svg)
        """
        if self.style == "plotly":
            self._plot_energy_usage_plotly(metrics, save_path)
        else:
            self._plot_energy_usage_matplotlib(metrics, save_path, format)
    
    def _plot_energy_usage_matplotlib(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot energy usage using matplotlib."""
        hardware_metrics = metrics.get("hardware", {})
        gpu_utilization = hardware_metrics.get("gpu_utilization", [])
        memory_usage = hardware_metrics.get("memory_usage_mb", [])
        temperature = hardware_metrics.get("gpu_temperature", [])
        
        if not any([gpu_utilization, memory_usage, temperature]):
            print("No hardware metrics available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # GPU Utilization
        if gpu_utilization:
            axes[0].plot(gpu_utilization)
            axes[0].set_title("GPU Utilization")
            axes[0].set_ylabel("Utilization (%)")
            axes[0].grid(True)
        
        # Memory Usage
        if memory_usage:
            axes[1].plot(memory_usage)
            axes[1].set_title("Memory Usage")
            axes[1].set_ylabel("Memory (MB)")
            axes[1].grid(True)
        
        # Temperature
        if temperature:
            axes[2].plot(temperature)
            axes[2].set_title("GPU Temperature")
            axes[2].set_ylabel("Temperature (°C)")
            axes[2].grid(True)
        
        # Energy summary
        energy_metrics = metrics.get("energy", {})
        if energy_metrics:
            total_energy = energy_metrics.get("total_energy_joules", 0)
            avg_power = energy_metrics.get("avg_power_watts", 0)
            
            axes[3].bar(["Total Energy", "Avg Power"], [total_energy, avg_power])
            axes[3].set_title("Energy Summary")
            axes[3].set_ylabel("Energy (J) / Power (W)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=format, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_energy_usage_plotly(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None
    ) -> None:
        """Plot energy usage using plotly."""
        hardware_metrics = metrics.get("hardware", {})
        gpu_utilization = hardware_metrics.get("gpu_utilization", [])
        memory_usage = hardware_metrics.get("memory_usage_mb", [])
        temperature = hardware_metrics.get("gpu_temperature", [])
        
        if not any([gpu_utilization, memory_usage, temperature]):
            print("No hardware metrics available for plotting")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("GPU Utilization", "Memory Usage", "GPU Temperature", "Energy Summary"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # GPU Utilization
        if gpu_utilization:
            fig.add_trace(
                go.Scatter(y=gpu_utilization, mode="lines", name="GPU Utilization"),
                row=1, col=1
            )
        
        # Memory Usage
        if memory_usage:
            fig.add_trace(
                go.Scatter(y=memory_usage, mode="lines", name="Memory Usage"),
                row=1, col=2
            )
        
        # Temperature
        if temperature:
            fig.add_trace(
                go.Scatter(y=temperature, mode="lines", name="Temperature"),
                row=2, col=1
            )
        
        # Energy summary
        energy_metrics = metrics.get("energy", {})
        if energy_metrics:
            total_energy = energy_metrics.get("total_energy_joules", 0)
            avg_power = energy_metrics.get("avg_power_watts", 0)
            
            fig.add_trace(
                go.Bar(x=["Total Energy", "Avg Power"], y=[total_energy, avg_power]),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Hardware and Energy Metrics",
            height=600,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_model_comparison(
        self, 
        results: List[Dict[str, Any]], 
        metric: str = "accuracy",
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot model comparison for a specific metric.
        
        Args:
            results: List of benchmark results
            metric: Metric to compare (accuracy, throughput, energy)
            save_path: Path to save plot
            format: Plot format (png, pdf, svg)
        """
        if self.style == "plotly":
            self._plot_model_comparison_plotly(results, metric, save_path)
        else:
            self._plot_model_comparison_matplotlib(results, metric, save_path, format)
    
    def _plot_model_comparison_matplotlib(
        self, 
        results: List[Dict[str, Any]], 
        metric: str,
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> None:
        """Plot model comparison using matplotlib."""
        if not results:
            return
        
        # Extract data
        labels = []
        values = []
        
        for result in results:
            config = result.get("config", {})
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            model_name = config.get("model", {}).get("name", "unknown")
            labels.append(f"{model_name}_{dataset_name}")
            
            if metric == "accuracy":
                value = result.get("test_results", {}).get("test_accuracy", 0)
            elif metric == "throughput":
                value = result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0)
            elif metric == "energy":
                value = result.get("metrics", {}).get("energy", {}).get("total_energy_joules", 0)
            else:
                value = 0
            
            values.append(value)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel("Model-Dataset Combination")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Comparison")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=format, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_model_comparison_plotly(
        self, 
        results: List[Dict[str, Any]], 
        metric: str,
        save_path: Optional[str] = None
    ) -> None:
        """Plot model comparison using plotly."""
        if not results:
            return
        
        # Extract data
        labels = []
        values = []
        
        for result in results:
            config = result.get("config", {})
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            model_name = config.get("model", {}).get("name", "unknown")
            labels.append(f"{model_name}_{dataset_name}")
            
            if metric == "accuracy":
                value = result.get("test_results", {}).get("test_accuracy", 0)
            elif metric == "throughput":
                value = result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0)
            elif metric == "energy":
                value = result.get("metrics", {}).get("energy", {}).get("total_energy_joules", 0)
            else:
                value = 0
            
            values.append(value)
        
        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, text=[f"{v:.3f}" for v in values], textposition='auto')
        ])
        
        fig.update_layout(
            title=f"{metric.capitalize()} Comparison",
            xaxis_title="Model-Dataset Combination",
            yaxis_title=metric.capitalize(),
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
