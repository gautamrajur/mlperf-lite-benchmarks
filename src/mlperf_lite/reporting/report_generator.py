"""Report generation for MLPerf Lite Benchmarks."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from mlperf_lite.reporting.plots import PlotGenerator


class ReportGenerator:
    """Generate comprehensive benchmark reports."""
    
    def __init__(
        self, 
        output_dir: Union[str, Path] = "reports",
        plot_style: str = "default"
    ) -> None:
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
            plot_style: Style for plots (default, seaborn, plotly)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_generator = PlotGenerator(style=plot_style)
    
    def generate_report(
        self, 
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        report_name: Optional[str] = None
    ) -> Path:
        """Generate comprehensive benchmark report.
        
        Args:
            results: Single result or list of results
            report_name: Name for the report
            
        Returns:
            Path to generated report
        """
        if isinstance(results, dict):
            results = [results]
        
        if report_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_name = f"benchmark_report_{timestamp}"
        
        report_dir = self.output_dir / report_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate different report formats
        self._generate_markdown_report(results, report_dir)
        self._generate_html_report(results, report_dir)
        self._generate_json_summary(results, report_dir)
        
        # Generate plots
        self._generate_plots(results, report_dir)
        
        # Generate comparison tables
        self._generate_comparison_tables(results, report_dir)
        
        return report_dir
    
    def _generate_markdown_report(
        self, 
        results: List[Dict[str, Any]], 
        report_dir: Path
    ) -> None:
        """Generate markdown report."""
        report_path = report_dir / "report.md"
        
        with open(report_path, 'w') as f:
            f.write("# MLPerf Lite Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Model | Dataset | Accuracy | Throughput | Energy |\n")
            f.write("|-------|---------|----------|------------|--------|\n")
            
            for result in results:
                config = result.get("config", {})
                model_name = config.get("model", {}).get("name", "unknown")
                dataset_name = config.get("dataset", {}).get("name", "unknown")
                
                test_acc = result.get("test_results", {}).get("test_accuracy", 0)
                throughput = result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0)
                energy = result.get("metrics", {}).get("energy", {}).get("total_energy_joules", 0)
                
                f.write(f"| {model_name} | {dataset_name} | {test_acc:.4f} | {throughput:.1f} | {energy:.1f} J |\n")
            
            f.write("\n")
            
            # Detailed results for each benchmark
            for i, result in enumerate(results):
                f.write(f"## Benchmark {i+1}: {result.get('config', {}).get('model', {}).get('name', 'Unknown')} on {result.get('config', {}).get('dataset', {}).get('name', 'Unknown')}\n\n")
                
                # Configuration
                f.write("### Configuration\n\n")
                config = result.get("config", {})
                f.write(f"- **Model**: {config.get('model', {}).get('name', 'unknown')}\n")
                f.write(f"- **Dataset**: {config.get('dataset', {}).get('name', 'unknown')}\n")
                f.write(f"- **Epochs**: {config.get('training', {}).get('epochs', 'unknown')}\n")
                f.write(f"- **Batch Size**: {config.get('dataset', {}).get('params', {}).get('batch_size', 'unknown')}\n")
                f.write(f"- **Device**: {config.get('benchmark', {}).get('device', 'unknown')}\n\n")
                
                # Results
                f.write("### Results\n\n")
                training_results = result.get("training_results", {})
                test_results = result.get("test_results", {})
                metrics = result.get("metrics", {})
                
                f.write(f"- **Best Validation Accuracy**: {training_results.get('best_val_accuracy', 0):.4f}\n")
                f.write(f"- **Test Accuracy**: {test_results.get('test_accuracy', 0):.4f}\n")
                f.write(f"- **Test Loss**: {test_results.get('test_loss', 0):.4f}\n")
                f.write(f"- **Training Time**: {training_results.get('training_time', 0):.2f} seconds\n")
                f.write(f"- **Epochs Completed**: {training_results.get('epochs_completed', 0)}\n")
                
                throughput = metrics.get("throughput", {})
                f.write(f"- **Throughput**: {throughput.get('samples_per_sec', 0):.2f} samples/sec\n")
                
                energy = metrics.get("energy", {})
                f.write(f"- **Total Energy**: {energy.get('total_energy_joules', 0):.2f} J\n")
                f.write(f"- **Average Power**: {energy.get('avg_power_watts', 0):.2f} W\n")
                
                model_info = metrics.get("model", {})
                f.write(f"- **Model Size**: {model_info.get('model_size_mb', 0):.2f} MB\n")
                f.write(f"- **Parameters**: {model_info.get('parameters_count', 0):,}\n")
                
                f.write("\n")
    
    def _generate_html_report(
        self, 
        results: List[Dict[str, Any]], 
        report_dir: Path
    ) -> None:
        """Generate HTML report."""
        report_path = report_dir / "report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MLPerf Lite Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ margin-bottom: 30px; }}
        .benchmark {{ margin-bottom: 40px; }}
        .metric {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>MLPerf Lite Benchmark Report</h1>
    <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Dataset</th>
                <th>Accuracy</th>
                <th>Throughput (samples/sec)</th>
                <th>Energy (J)</th>
                <th>Training Time (s)</th>
            </tr>
"""
        
        for result in results:
            config = result.get("config", {})
            model_name = config.get("model", {}).get("name", "unknown")
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            
            test_acc = result.get("test_results", {}).get("test_accuracy", 0)
            throughput = result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0)
            energy = result.get("metrics", {}).get("energy", {}).get("total_energy_joules", 0)
            training_time = result.get("training_results", {}).get("training_time", 0)
            
            html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{dataset_name}</td>
                <td>{test_acc:.4f}</td>
                <td>{throughput:.1f}</td>
                <td>{energy:.1f}</td>
                <td>{training_time:.1f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
"""
        
        # Detailed results
        for i, result in enumerate(results):
            config = result.get("config", {})
            model_name = config.get("model", {}).get("name", "unknown")
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            
            training_results = result.get("training_results", {})
            test_results = result.get("test_results", {})
            metrics = result.get("metrics", {})
            
            html_content += f"""
    <div class="benchmark">
        <h2>Benchmark {i+1}: {model_name} on {dataset_name}</h2>
        
        <h3>Configuration</h3>
        <div class="metric">
            <strong>Model:</strong> {model_name}<br>
            <strong>Dataset:</strong> {dataset_name}<br>
            <strong>Epochs:</strong> {config.get('training', {}).get('epochs', 'unknown')}<br>
            <strong>Batch Size:</strong> {config.get('dataset', {}).get('params', {}).get('batch_size', 'unknown')}<br>
            <strong>Device:</strong> {config.get('benchmark', {}).get('device', 'unknown')}
        </div>
        
        <h3>Results</h3>
        <div class="metric">
            <strong>Best Validation Accuracy:</strong> {training_results.get('best_val_accuracy', 0):.4f}<br>
            <strong>Test Accuracy:</strong> {test_results.get('test_accuracy', 0):.4f}<br>
            <strong>Test Loss:</strong> {test_results.get('test_loss', 0):.4f}<br>
            <strong>Training Time:</strong> {training_results.get('training_time', 0):.2f} seconds<br>
            <strong>Epochs Completed:</strong> {training_results.get('epochs_completed', 0)}<br>
            <strong>Throughput:</strong> {metrics.get('throughput', {}).get('samples_per_sec', 0):.2f} samples/sec<br>
            <strong>Total Energy:</strong> {metrics.get('energy', {}).get('total_energy_joules', 0):.2f} J<br>
            <strong>Average Power:</strong> {metrics.get('energy', {}).get('avg_power_watts', 0):.2f} W<br>
            <strong>Model Size:</strong> {metrics.get('model', {}).get('model_size_mb', 0):.2f} MB<br>
            <strong>Parameters:</strong> {metrics.get('model', {}).get('parameters_count', 0):,}
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    def _generate_json_summary(
        self, 
        results: List[Dict[str, Any]], 
        report_dir: Path
    ) -> None:
        """Generate JSON summary."""
        summary_path = report_dir / "summary.json"
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_benchmarks": len(results),
            "summary": []
        }
        
        for result in results:
            config = result.get("config", {})
            model_name = config.get("model", {}).get("name", "unknown")
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            
            summary_item = {
                "model": model_name,
                "dataset": dataset_name,
                "test_accuracy": result.get("test_results", {}).get("test_accuracy", 0),
                "throughput_samples_per_sec": result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0),
                "total_energy_joules": result.get("metrics", {}).get("energy", {}).get("total_energy_joules", 0),
                "training_time_seconds": result.get("training_results", {}).get("training_time", 0),
                "model_size_mb": result.get("metrics", {}).get("model", {}).get("model_size_mb", 0),
                "parameters_count": result.get("metrics", {}).get("model", {}).get("parameters_count", 0),
            }
            
            summary["summary"].append(summary_item)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_plots(
        self, 
        results: List[Dict[str, Any]], 
        report_dir: Path
    ) -> None:
        """Generate plots."""
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Training curves for each result
        for i, result in enumerate(results):
            metrics = result.get("metrics", {})
            if metrics:
                plot_path = plots_dir / f"training_curves_{i+1}.png"
                self.plot_generator.plot_training_curves(metrics, str(plot_path))
        
        # Comparison plots
        if len(results) > 1:
            # Throughput comparison
            throughput_plot = plots_dir / "throughput_comparison.png"
            self.plot_generator.plot_throughput_comparison(results, str(throughput_plot))
            
            # Accuracy comparison
            accuracy_plot = plots_dir / "accuracy_comparison.png"
            self.plot_generator.plot_model_comparison(results, "accuracy", str(accuracy_plot))
            
            # Energy comparison
            energy_plot = plots_dir / "energy_comparison.png"
            self.plot_generator.plot_model_comparison(results, "energy", str(energy_plot))
        
        # Energy usage plots for each result
        for i, result in enumerate(results):
            metrics = result.get("metrics", {})
            if metrics:
                energy_plot = plots_dir / f"energy_usage_{i+1}.png"
                self.plot_generator.plot_energy_usage(metrics, str(energy_plot))
    
    def _generate_comparison_tables(
        self, 
        results: List[Dict[str, Any]], 
        report_dir: Path
    ) -> None:
        """Generate comparison tables."""
        tables_dir = report_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        # Create comparison DataFrame
        data = []
        for result in results:
            config = result.get("config", {})
            model_name = config.get("model", {}).get("name", "unknown")
            dataset_name = config.get("dataset", {}).get("name", "unknown")
            
            row = {
                "Model": model_name,
                "Dataset": dataset_name,
                "Test_Accuracy": result.get("test_results", {}).get("test_accuracy", 0),
                "Throughput_Samples_Per_Sec": result.get("metrics", {}).get("throughput", {}).get("samples_per_sec", 0),
                "Total_Energy_Joules": result.get("metrics", {}).get("energy", {}).get("total_energy_joules", 0),
                "Training_Time_Seconds": result.get("training_results", {}).get("training_time", 0),
                "Model_Size_MB": result.get("metrics", {}).get("model", {}).get("model_size_mb", 0),
                "Parameters_Count": result.get("metrics", {}).get("model", {}).get("parameters_count", 0),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = tables_dir / "comparison.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as Excel
        excel_path = tables_dir / "comparison.xlsx"
        df.to_excel(excel_path, index=False)
        
        # Generate summary statistics
        summary_stats = df.describe()
        stats_path = tables_dir / "summary_statistics.csv"
        summary_stats.to_csv(stats_path)
    
    def generate_single_benchmark_report(
        self, 
        result: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> Path:
        """Generate report for a single benchmark result.
        
        Args:
            result: Single benchmark result
            report_name: Name for the report
            
        Returns:
            Path to generated report
        """
        return self.generate_report([result], report_name)
