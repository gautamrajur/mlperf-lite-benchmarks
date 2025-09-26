"""Energy monitoring for MLPerf Lite Benchmarks."""

import time
import threading
from typing import Any, Dict, List, Optional
import subprocess
import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class EnergyMonitor:
    """Monitor energy consumption using NVIDIA SMI and NVML."""
    
    def __init__(self, device: str = "cuda", gpu_id: int = 0) -> None:
        """Initialize energy monitor.
        
        Args:
            device: Device to monitor
            gpu_id: GPU ID to monitor
        """
        self.device = device
        self.gpu_id = gpu_id
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Energy data
        self.power_readings: List[float] = []
        self.timestamps: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
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
    
    def start_monitoring(self, interval: float = 0.1) -> None:
        """Start energy monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.power_readings = []
        self.timestamps = []
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop energy monitoring and return results.
        
        Returns:
            Dictionary with energy metrics
        """
        if not self.monitoring:
            return {"total_energy": 0.0, "avg_power": 0.0, "peak_power": 0.0}
        
        self.monitoring = False
        self.end_time = time.time()
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate energy metrics
        total_time = self.end_time - self.start_time if self.start_time else 0.0
        
        if not self.power_readings:
            return {"total_energy": 0.0, "avg_power": 0.0, "peak_power": 0.0}
        
        # Calculate total energy (integrate power over time)
        total_energy = 0.0
        for i in range(1, len(self.power_readings)):
            dt = self.timestamps[i] - self.timestamps[i-1]
            avg_power = (self.power_readings[i] + self.power_readings[i-1]) / 2
            total_energy += avg_power * dt
        
        avg_power = sum(self.power_readings) / len(self.power_readings)
        peak_power = max(self.power_readings)
        
        return {
            "total_energy": total_energy,
            "avg_power": avg_power,
            "peak_power": peak_power,
            "total_time": total_time,
            "num_readings": len(self.power_readings)
        }
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                power = self._get_current_power()
                if power is not None:
                    self.power_readings.append(power)
                    self.timestamps.append(time.time())
            except Exception as e:
                print(f"Warning: Power reading failed: {e}")
            
            time.sleep(interval)
    
    def _get_current_power(self) -> Optional[float]:
        """Get current power consumption.
        
        Returns:
            Current power in watts, or None if unavailable
        """
        # Try NVML first
        if self.nvml_available:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                return power / 1000.0  # Convert mW to W
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if result.returncode == 0:
                power_str = result.stdout.strip().split('\n')[self.gpu_id]
                return float(power_str)
        except Exception:
            pass
        
        return None
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information.
        
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
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                
                info.update({
                    "gpu_name": name,
                    "memory_total_mb": memory_info.total / (1024 * 1024),
                    "memory_free_mb": memory_info.free / (1024 * 1024),
                    "memory_used_mb": memory_info.used / (1024 * 1024),
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "temperature": temperature,
                })
            except Exception as e:
                info["error"] = str(e)
        
        return info
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current energy and performance metrics.
        
        Returns:
            Dictionary with current metrics
        """
        metrics = {
            "timestamp": time.time(),
            "monitoring": self.monitoring,
        }
        
        # Get current power
        current_power = self._get_current_power()
        if current_power is not None:
            metrics["current_power_watts"] = current_power
        
        # Get GPU info
        gpu_info = self.get_gpu_info()
        metrics.update(gpu_info)
        
        return metrics


class CPUMonitor:
    """Monitor CPU energy consumption."""
    
    def __init__(self) -> None:
        """Initialize CPU monitor."""
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.cpu_readings: List[float] = []
        self.timestamps: List[float] = []
        self.start_time: Optional[float] = None
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start CPU monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.cpu_readings = []
        self.timestamps = []
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop CPU monitoring and return results.
        
        Returns:
            Dictionary with CPU metrics
        """
        if not self.monitoring:
            return {"avg_cpu_percent": 0.0, "peak_cpu_percent": 0.0}
        
        self.monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.cpu_readings:
            return {"avg_cpu_percent": 0.0, "peak_cpu_percent": 0.0}
        
        avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings)
        peak_cpu = max(self.cpu_readings)
        
        return {
            "avg_cpu_percent": avg_cpu,
            "peak_cpu_percent": peak_cpu,
            "num_readings": len(self.cpu_readings)
        }
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_readings.append(cpu_percent)
                self.timestamps.append(time.time())
            except Exception as e:
                print(f"Warning: CPU reading failed: {e}")
            
            time.sleep(interval)
