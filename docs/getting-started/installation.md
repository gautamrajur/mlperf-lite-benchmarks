# Installation

## Requirements

- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU support)
- NVIDIA drivers compatible with your CUDA version

## Installation Methods

### From PyPI (Recommended)

```bash
pip install mlperf-lite-benchmarks
```

### From Source

```bash
# Clone the repository
git clone https://github.com/mlperf-lite/mlperf-lite-benchmarks.git
cd mlperf-lite-benchmarks

# Install in development mode
pip install -e ".[dev]"

# Or install with all dependencies
pip install -e .
```

### Development Installation

For development work, install with development dependencies:

```bash
git clone https://github.com/mlperf-lite/mlperf-lite-benchmarks.git
cd mlperf-lite-benchmarks
pip install -e ".[dev,docs]"
```

## Verify Installation

Check that the installation was successful:

```bash
# Check CLI is available
mlperf-lite --help

# Check system information
mlperf-lite system-info

# Run a quick test
python -c "import mlperf_lite; print('Installation successful!')"
```

## Optional Dependencies

### GPU Monitoring

For advanced GPU monitoring features:

```bash
pip install pynvml
```

### Enhanced Plotting

For interactive plots:

```bash
pip install plotly
```

### Documentation

For building documentation:

```bash
pip install mkdocs mkdocs-material
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related issues:

1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Check PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Memory Issues

For systems with limited memory:

1. Reduce batch size in configuration files
2. Use gradient accumulation instead of large batches
3. Enable mixed precision training

### Permission Issues

If you encounter permission issues:

```bash
# Use user installation
pip install --user mlperf-lite-benchmarks

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install mlperf-lite-benchmarks
```
