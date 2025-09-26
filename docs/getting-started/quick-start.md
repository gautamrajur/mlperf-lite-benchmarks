# Quick Start

This guide will help you run your first benchmark with MLPerf Lite Benchmarks.

## Basic Usage

### Run a Default Benchmark

The simplest way to get started is to run a benchmark with default settings:

```bash
mlperf-lite run
```

This will run a ResNet-18 model on CIFAR-10 dataset with default configuration.

### Run with Custom Configuration

Create a custom configuration file:

```bash
# Create a configuration from template
mlperf-lite create-config resnet_cifar10 --output my_config.yaml

# Run with custom configuration
mlperf-lite run --config my_config.yaml
```

### Run Inference-Only Benchmark

For inference-only benchmarking:

```bash
mlperf-lite inference --config my_config.yaml --num-batches 100
```

## Configuration Examples

### ResNet on CIFAR-10

```yaml
dataset:
  name: cifar10
  params:
    batch_size: 128
    download: true

model:
  name: resnet
  params:
    architecture: resnet18
    learning_rate: 0.1

training:
  epochs: 100
  target_accuracy: 0.95
```

### BERT on SST-2

```yaml
dataset:
  name: sst2
  params:
    batch_size: 32
    model_name: bert-base-uncased

model:
  name: bert
  params:
    learning_rate: 2e-5

training:
  epochs: 3
  target_accuracy: 0.90
```

## Understanding Results

After running a benchmark, you'll find results in the output directory:

```
results/
├── results.json          # Detailed results
├── config.json          # Configuration used
├── checkpoints/         # Model checkpoints
└── logs/               # Training logs
```

### Key Metrics

- **Accuracy**: Final test accuracy and best validation accuracy
- **Throughput**: Samples processed per second
- **Energy**: Total energy consumption in Joules
- **Training Time**: Total training time in seconds
- **Model Size**: Model size in MB and parameter count

## Generating Reports

Generate comprehensive reports from your results:

```bash
# Generate report from results directory
mlperf-lite report --results-dir results/ --output reports/

# Generate comparison report from multiple runs
mlperf-lite report --results-dir results/ --compare --output reports/
```

## Command Line Options

### Run Command

```bash
mlperf-lite run [OPTIONS]

Options:
  -c, --config PATH        Configuration file path
  -o, --output-dir PATH    Output directory [default: results]
  -d, --device TEXT        Device to use (cuda/cpu) [default: cuda]
  -v, --verbose            Verbose output
```

### Inference Command

```bash
mlperf-lite inference [OPTIONS]

Options:
  -c, --config PATH        Configuration file path
  -o, --output-dir PATH    Output directory [default: results]
  -d, --device TEXT        Device to use (cuda/cpu) [default: cuda]
  -n, --num-batches INT    Number of batches to run [default: 100]
  -w, --warmup-batches INT Number of warmup batches [default: 10]
```

### System Info Command

```bash
mlperf-lite system-info
```

## Next Steps

Now that you've run your first benchmark:

1. **Explore Different Models**: Try different architectures and datasets
2. **Customize Configurations**: Modify hyperparameters and training settings
3. **Compare Results**: Run multiple benchmarks and compare performance
4. **Generate Reports**: Create detailed reports with plots and analysis
5. **Monitor Resources**: Use energy and hardware monitoring features

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size in configuration
- Use gradient accumulation
- Enable mixed precision training

**Slow Training**:
- Check GPU utilization with `nvidia-smi`
- Increase number of data loader workers
- Use appropriate batch size for your GPU

**Configuration Errors**:
- Validate configuration with `mlperf-lite validate-config`
- Check required fields in configuration
- Use provided templates as starting points

### Getting Help

- Check the [User Guide](user-guide/) for detailed documentation
- Look at [Examples](examples/) for configuration templates
- Report issues on [GitHub](https://github.com/mlperf-lite/mlperf-lite-benchmarks/issues)
