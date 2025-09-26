"""System information utilities for MLPerf Lite Benchmarks."""

import os
import platform
import subprocess
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


class SystemInfo:
    """Collect and provide system information."""
    
    def __init__(self) -> None:
        """Initialize system info collector."""
        self._system_info: Optional[Dict[str, Any]] = None
        self._gpu_info: Optional[List[Dict[str, Any]]] = None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        if self._system_info is None:
            self._system_info = self._collect_system_info()
        return self._system_info
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information.
        
        Returns:
            List of GPU information dictionaries
        """
        if self._gpu_info is None:
            self._gpu_info = self._collect_gpu_info()
        return self._gpu_info
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information.
        
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
        
        # Add NVIDIA driver information
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
    
    def _collect_gpu_info(self) -> List[Dict[str, Any]]:
        """Collect GPU information.
        
        Returns:
            List of GPU information dictionaries
        """
        gpu_info = []
        
        # Try PyTorch first
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count,
                    "max_threads_per_block": props.max_threads_per_block,
                    "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
                    "warp_size": props.warp_size,
                    "source": "pytorch",
                })
        
        # Try NVML for additional information
        elif NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    props = pynvml.nvmlDeviceGetProperties(handle)
                    
                    gpu_info.append({
                        "id": i,
                        "name": name,
                        "memory_total_gb": memory_info.total / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multiProcessorCount,
                        "max_threads_per_block": props.maxThreadsPerBlock,
                        "max_threads_per_multiprocessor": props.maxThreadsPerMultiProcessor,
                        "warp_size": props.warpSize,
                        "source": "nvml",
                    })
            except Exception as e:
                print(f"Warning: NVML GPU info collection failed: {e}")
        
        # Fallback to nvidia-smi
        if not gpu_info:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 3:
                                gpu_info.append({
                                    "id": int(parts[0]),
                                    "name": parts[1],
                                    "memory_total_gb": float(parts[2]) / 1024,
                                    "source": "nvidia-smi",
                                })
            except Exception as e:
                print(f"Warning: nvidia-smi GPU info collection failed: {e}")
        
        return gpu_info
    
    def get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA-specific information.
        
        Returns:
            Dictionary with CUDA information
        """
        cuda_info = {
            "available": False,
            "version": None,
            "device_count": 0,
            "devices": [],
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            cuda_info.update({
                "available": True,
                "version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": self.get_gpu_info(),
            })
        
        return cuda_info
    
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
            "CUDA_CACHE_DISABLE",
            "CUDA_LAUNCH_BLOCKING",
        ]
        
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return {
            "environment_variables": env_vars,
            "working_directory": os.getcwd(),
            "user": os.getenv("USER", "unknown"),
            "path": os.getenv("PATH", "").split(os.pathsep),
        }
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture-specific information.
        
        Returns:
            Dictionary with architecture information
        """
        arch_info = {
            "cpu_architecture": platform.machine(),
            "system_architecture": platform.architecture()[0],
            "gpu_architectures": [],
        }
        
        # Get GPU architectures
        for gpu in self.get_gpu_info():
            if "compute_capability" in gpu:
                arch_info["gpu_architectures"].append({
                    "gpu_id": gpu["id"],
                    "name": gpu["name"],
                    "compute_capability": gpu["compute_capability"],
                })
        
        return arch_info
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance-related information.
        
        Returns:
            Dictionary with performance information
        """
        perf_info = {
            "cpu": {
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
            },
            "gpu": [],
        }
        
        # Add GPU performance info
        for gpu in self.get_gpu_info():
            perf_info["gpu"].append({
                "id": gpu["id"],
                "name": gpu["name"],
                "memory_total_gb": gpu["memory_total_gb"],
                "compute_capability": gpu.get("compute_capability"),
            })
        
        return perf_info
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of system information.
        
        Returns:
            Dictionary with system summary
        """
        system_info = self.get_system_info()
        gpu_info = self.get_gpu_info()
        
        return {
            "platform": f"{system_info['platform']['system']} {system_info['platform']['release']}",
            "python_version": system_info['platform']['python_version'],
            "cpu_count": system_info['cpu']['count'],
            "memory_total_gb": system_info['memory']['total_gb'],
            "gpu_count": len(gpu_info),
            "gpu_names": [gpu['name'] for gpu in gpu_info],
            "pytorch_available": TORCH_AVAILABLE,
            "cuda_available": system_info.get('pytorch', {}).get('cuda_available', False),
            "cuda_version": system_info.get('pytorch', {}).get('cuda_version'),
        }
