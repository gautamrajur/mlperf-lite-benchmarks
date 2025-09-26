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

### Installation

```bash
# Clone the repository
git clone https://github.com/mlperf-lite/mlperf-lite-benchmarks.git
cd mlperf-lite-benchmarks

# Install in development mode
pip install -e ".[dev]"

# Or install with all dependencies
pip install -e .
```

### Basic Usage

```bash
# Run a benchmark with default configuration
mlperf-lite run --config configs/resnet_cifar10.yaml

# Run with custom parameters
mlperf-lite run --model resnet --dataset cifar10 --epochs 100 --batch-size 128

# Generate reports
mlperf-lite report --results-dir results/ --output reports/
```

## Project Structure

```
mlperf-lite-benchmarks/
├── src/mlperf_lite/          # Main package
│   ├── datasets/             # Dataset loaders
│   ├── models/               # Model wrappers
│   ├── benchmarks/           # Benchmarking harness
│   ├── monitoring/           # Energy and performance monitoring
│   ├── reporting/            # Report generation
│   └── utils/                # Utilities
├── benchmarks/               # Benchmark configurations
├── configs/                  # YAML/JSON configuration files
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── scripts/                  # Helper scripts
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mlperf_lite

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m gpu         # Run GPU tests only
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.
