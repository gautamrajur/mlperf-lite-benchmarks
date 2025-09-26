# MLPerf Lite Benchmarks

A production-grade MLPerf-style benchmarking framework for ResNet, BERT, and UNet models.

## Features

- **Model Support**: ResNet, BERT, and UNet architectures
- **Dataset Support**: CIFAR-10, SST-2, and custom dataset loader interface
- **Metrics Collection**: Throughput, training time-to-accuracy, energy usage
- **Multi-GPU Support**: Benchmarking across different GPU architectures (Turing, Ampere)
- **CUDA Version Comparison**: Support for different CUDA versions
- **Regression Detection**: Kernel-level regression detection and architecture-specific performance monitoring
- **Config-Driven**: YAML/JSON configuration for experiment orchestration
- **Rich Reporting**: Markdown/HTML reports with plots and summaries

## Quick Start

```bash
# Install the package
pip install mlperf-lite-benchmarks

# Run a benchmark with default configuration
mlperf-lite run

# Run with custom configuration
mlperf-lite run --config configs/resnet_cifar10.yaml

# Generate reports
mlperf-lite report --results-dir results/ --output reports/
```

## Architecture

The framework is designed with modularity and extensibility in mind:

- **Dataset Loaders**: Pluggable dataset interfaces for easy extension
- **Model Wrappers**: Standardized model interfaces with common training patterns
- **Benchmarking Harness**: Comprehensive metrics collection and monitoring
- **Energy Monitoring**: NVIDIA SMI and NVML integration for power consumption tracking
- **Configuration System**: YAML/JSON-driven experiment orchestration
- **Report Generation**: Rich reports with plots, tables, and summaries

## Supported Models

| Model | Architecture | Dataset | Task |
|-------|-------------|---------|------|
| ResNet | CNN | CIFAR-10 | Image Classification |
| BERT | Transformer | SST-2 | Text Classification |
| UNet | CNN | Custom | Semantic Segmentation |

## Key Metrics

- **Throughput**: Samples/second, tokens/second
- **Accuracy**: Training and validation accuracy curves
- **Energy**: Total energy consumption and average power
- **Hardware**: GPU utilization, memory usage, temperature
- **Model**: Parameter count, model size, FLOPs

## Getting Started

Ready to start benchmarking? Check out our [Installation Guide](getting-started/installation.md) and [Quick Start Tutorial](getting-started/quick-start.md).

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing/development.md) for details on how to get started.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
