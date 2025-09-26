"""Hardware monitoring for MLPerf Lite Benchmarks."""

import time
import threading
from typing import Any, Dict, List, Optional
import psutil
import subprocess

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class HardwareMonitor:
    """Monitor hardware utilization and performance."""
    
    def __init__(self, device: str = "cuda", gpu_id: int = 0) -> None:
        """Initialize hardware monitor.
        
        Args:
            device: Device to monitor
            gpu_id: GPU ID to monitor
        """
        self.device = device
        self.gpu_id = gpu_id
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Monitoring data
        self.gpu_utilization: List[float] = []
        self.memory_usage: List[float] = []
        self.temperature: List[float] = []
        self.timestamps: List[float] = []
        
        # Initialize NVML if available
        if NVML_AVAILABLE and device == "cuda":
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.nvml_available = True
            except Exception as e:
                print(f"Warning: NVML initialization failed: {e}")
                self.nvml_available = False
        else:
            self.nvml_available = False
    
    def start_monitoring(self, interval: float = 0.5) -> None:
        """Start hardware monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.gpu_utilization = []
        self.memory_usage = []
        self.temperature = []
        self.timestamps = []
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop hardware monitoring and return results.
        
        Returns:
            Dictionary with hardware metrics
        """
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate statistics
        results = {
            "num_readings": len(self.timestamps),
            "duration": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0,
        }
        
        if self.gpu_utilization:
            results.update({
                "gpu_utilization": {
                    "avg": sum(self.gpu_utilization) / len(self.gpu_utilization),
                    "min": min(self.gpu_utilization),
                    "max": max(self.gpu_utilization),
                    "values": self.gpu_utilization.copy(),
                }
            })
        
        if self.memory_usage:
            results.update({
                "memory_usage": {
                    "avg_mb": sum(self.memory_usage) / len(self.memory_usage),
                    "min_mb": min(self.memory_usage),
                    "max_mb": max(self.memory_usage),
                    "values": self.memory_usage.copy(),
                }
            })
        
        if self.temperature:
            results.update({
                "temperature": {
                    "avg_celsius": sum(self.temperature) / len(self.temperature),
                    "min_celsius": min(self.temperature),
                    "max_celsius": max(self.temperature),
                    "values": self.temperature.copy(),
                }
            })
        
        return results
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._get_current_metrics()
                if metrics:
                    self.gpu_utilization.append(metrics.get("gpu_utilization", 0.0))
                    self.memory_usage.append(metrics.get("memory_usage_mb", 0.0))
                    self.temperature.append(metrics.get("temperature", 0.0))
                    self.timestamps.append(time.time())
            except Exception as e:
                print(f"Warning: Hardware monitoring failed: {e}")
            
            time.sleep(interval)
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current hardware metrics.
        
        Returns:
            Dictionary with current metrics
        """
        metrics = {}
        
        # Try NVML first
        if self.nvml_available:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                
                metrics.update({
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "memory_usage_mb": memory_info.used / (1024 * 1024),
                    "memory_total_mb": memory_info.total / (1024 * 1024),
                    "temperature": temperature,
                })
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        if not metrics:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                if result.returncode == 0:
                    values = result.stdout.strip().split('\n')[self.gpu_id].split(', ')
                    if len(values) >= 4:
                        metrics.update({
                            "gpu_utilization": float(values[0]),
                            "memory_usage_mb": float(values[1]),
                            "memory_total_mb": float(values[2]),
                            "temperature": float(values[3]),
                        })
            except Exception:
                pass
        
        return metrics
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information.
        
        Returns:
            Dictionary with GPU information
        """
        info = {
            "gpu_id": self.gpu_id,
            "device": self.device,
            "nvml_available": self.nvml_available,
        }
        
        if self.nvml_available:
            try:
                name = pynvml.nvmlDeviceGetName(self.handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_memory = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
                
                info.update({
                    "gpu_name": name,
                    "memory_total_mb": memory_info.total / (1024 * 1024),
                    "memory_free_mb": memory_info.free / (1024 * 1024),
                    "memory_used_mb": memory_info.used / (1024 * 1024),
                    "graphics_clock_mhz": clock_graphics,
                    "memory_clock_mhz": clock_memory,
                    "power_limit_watts": power_limit[1] / 1000.0,  # Convert mW to W
                })
            except Exception as e:
                info["error"] = str(e)
        
        return info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.
        
        Returns:
            Dictionary with system information
        """
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage": psutil.disk_usage('/')._asdict(),
        }


class PerformanceProfiler:
    """Profile performance metrics during training/inference."""
    
    def __init__(self, device: str = "cuda") -> None:
        """Initialize performance profiler.
        
        Args:
            device: Device to profile
        """
        self.device = device
        self.profiles: List[Dict[str, Any]] = []
    
    def start_profile(self, name: str) -> None:
        """Start profiling a named operation.
        
        Args:
            name: Name of the operation to profile
        """
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()
        
        self.profiles.append({
            "name": name,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
        })
    
    def end_profile(self, name: str) -> Dict[str, Any]:
        """End profiling a named operation.
        
        Args:
            name: Name of the operation
            
        Returns:
            Profile data
        """
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Find the matching start profile
        for i, profile in enumerate(self.profiles):
            if profile["name"] == name and "end_time" not in profile:
                profile.update({
                    "end_time": end_time,
                    "end_memory": end_memory,
                    "duration": end_time - profile["start_time"],
                    "memory_delta": end_memory - profile["start_memory"],
                })
                return profile
        
        return {}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device == "cuda":
            try:
                import torch
                return torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        
        # Fallback to system memory
        return psutil.virtual_memory().used / (1024 * 1024)
    
    def get_profiles(self) -> List[Dict[str, Any]]:
        """Get all profile data.
        
        Returns:
            List of profile data
        """
        return self.profiles.copy()
    
    def clear_profiles(self) -> None:
        """Clear all profile data."""
        self.profiles.clear()
