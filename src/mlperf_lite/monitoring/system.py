"""System monitoring for MLPerf Lite Benchmarks."""

import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class SystemMonitor:
    """Monitor system-wide metrics and information."""
    
    def __init__(self) -> None:
        """Initialize system monitor."""
        self.start_time = time.time()
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            "cpu": {
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / (1024**3),
                "used_gb": psutil.disk_usage('/').used / (1024**3),
                "free_gb": psutil.disk_usage('/').free / (1024**3),
                "percent": psutil.disk_usage('/').percent,
            },
        }
        
        # Add PyTorch information
        if TORCH_AVAILABLE:
            info["pytorch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
            
            # Add GPU information
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_info.append({
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
                    })
                info["gpus"] = gpu_info
        
        # Add NVML information
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                info["nvidia"] = {
                    "driver_version": driver_version,
                    "nvml_version": pynvml.nvmlSystemGetNVMLVersion().decode('utf-8'),
                }
            except Exception as e:
                info["nvidia"] = {"error": str(e)}
        
        return info
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.
        
        Returns:
            Dictionary with current metrics
        """
        return {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_per_core": psutil.cpu_percent(interval=None, percpu=True),
            "memory": {
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "used_gb": psutil.disk_usage('/').used / (1024**3),
                "free_gb": psutil.disk_usage('/').free / (1024**3),
                "percent": psutil.disk_usage('/').percent,
            },
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
        }
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network interface information.
        
        Returns:
            Dictionary with network information
        """
        try:
            net_io = psutil.net_io_counters()
            net_connections = psutil.net_connections()
            
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "active_connections": len(net_connections),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information.
        
        Returns:
            Dictionary with process information
        """
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "name": process.name(),
                "cpu_percent": process.cpu_percent(),
                "memory_info": process.memory_info()._asdict(),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment variables and configuration.
        
        Returns:
            Dictionary with environment information
        """
        env_vars = {}
        relevant_vars = [
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TORCH_CUDNN_V8_API_ENABLED",
        ]
        
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return {
            "environment_variables": env_vars,
            "working_directory": os.getcwd(),
            "user": os.getenv("USER", "unknown"),
        }
    
    def benchmark_system(self, duration: float = 10.0) -> Dict[str, Any]:
        """Run a quick system benchmark.
        
        Args:
            duration: Duration of benchmark in seconds
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Running system benchmark for {duration} seconds...")
        
        # CPU benchmark
        start_time = time.time()
        cpu_count = psutil.cpu_count()
        
        def cpu_intensive_task():
            result = 0
            for i in range(1000000):
                result += i * i
            return result
        
        # Run CPU benchmark
        cpu_start = time.time()
        for _ in range(cpu_count):
            cpu_intensive_task()
        cpu_time = time.time() - cpu_start
        
        # Memory benchmark
        memory_start = time.time()
        data = []
        for i in range(1000000):
            data.append(i)
        memory_time = time.time() - memory_start
        
        # Disk benchmark
        disk_start = time.time()
        test_file = "/tmp/mlperf_test.txt"
        with open(test_file, 'w') as f:
            for i in range(100000):
                f.write(f"Line {i}\n")
        disk_write_time = time.time() - disk_start
        
        disk_start = time.time()
        with open(test_file, 'r') as f:
            _ = f.read()
        disk_read_time = time.time() - disk_start
        
        # Cleanup
        try:
            os.remove(test_file)
        except Exception:
            pass
        
        total_time = time.time() - start_time
        
        return {
            "duration": total_time,
            "cpu_benchmark": {
                "time_seconds": cpu_time,
                "operations_per_second": cpu_count / cpu_time,
            },
            "memory_benchmark": {
                "time_seconds": memory_time,
                "allocations_per_second": 1000000 / memory_time,
            },
            "disk_benchmark": {
                "write_time_seconds": disk_write_time,
                "read_time_seconds": disk_read_time,
                "write_mb_per_sec": (100000 * 10) / (disk_write_time * 1024 * 1024),  # Approximate
                "read_mb_per_sec": (100000 * 10) / (disk_read_time * 1024 * 1024),  # Approximate
            },
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive system report.
        
        Returns:
            Dictionary with comprehensive system information
        """
        return {
            "system_info": self.system_info,
            "current_metrics": self.get_current_metrics(),
            "network_info": self.get_network_info(),
            "process_info": self.get_process_info(),
            "environment_info": self.get_environment_info(),
            "timestamp": time.time(),
        }
