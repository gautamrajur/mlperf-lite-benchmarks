"""MLPerf Lite Benchmarks - Production-grade benchmarking framework."""

__version__ = "0.1.0"
__author__ = "MLPerf Lite Team"
__email__ = "team@mlperf-lite.org"

from mlperf_lite.benchmarks import BenchmarkRunner
from mlperf_lite.models import ModelFactory
from mlperf_lite.datasets import DatasetFactory

__all__ = [
    "BenchmarkRunner",
    "ModelFactory", 
    "DatasetFactory",
    "__version__",
    "__author__",
    "__email__",
]
