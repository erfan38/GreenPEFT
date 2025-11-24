import pynvml
import psutil
import time
import threading
from collections import deque
from typing import Dict, Optional, List
import json

class EnergyMetrics:
    def __init__(self):
        self.total_energy_wh: float = 0.0
        self.current_power_w: float = 0.0
        self.budget_used_percent: float = 0.0
        self.gpu_energy_wh: float = 0.0
        self.cpu_energy_wh: float = 0.0
        self.timestamp: float = time.time()

class EnergyMonitor:
    """Real-time energy monitoring for PEFT training"""
    
    def __init__(self, energy_budget_wh: float = 100.0, sampling_interval: float = 1.0):
        self.energy_budget_wh = energy_budget_wh
        self.sampling_interval = sampling_interval
        
        # Initialize NVIDIA ML
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
        except Exception:
            self.gpu_count = 0
            self.gpu_handles = []
            print("Warning: NVIDIA GPU monitoring not available")
        
        # Energy tracking
        self.start_time = time.time()
        self.last_update = self.start_time
        self.total_energy_wh = 0.0
        
        # Power history for moving averages
        self.power_history = deque(maxlen=60)  # Last 60 measurements
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start background energy monitoring"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> EnergyMetrics:
        """Stop monitoring and return final metrics"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        return self.get_current_metrics()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            self._update_energy_consumption()
            time.sleep(self.sampling_interval)
            
    def _get_gpu_power(self) -> float:
        """Get total GPU power consumption in watts"""
        total_power = 0.0
        for handle in self.gpu_handles:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                total_power += power_mw / 1000.0  # Convert mW to W
            except Exception:
                continue
        return total_power
    
    def _get_cpu_power(self) -> float:
        """Estimate CPU power consumption"""
        cpu_percent = psutil.cpu_percent()
        # Rough estimate: 100W max CPU power
        return (cpu_percent / 100.0) * 100.0
    
    def _update_energy_consumption(self):
        """Update energy consumption calculations"""
        current_time = time.time()
        time_delta = current_time - self.last_update
        
        # Get current power
        gpu_power = self._get_gpu_power()
        cpu_power = self._get_cpu_power()
        total_power = gpu_power + cpu_power
        
        # Calculate energy (Power × Time in hours)
        energy_increment = total_power * (time_delta / 3600.0)
        self.total_energy_wh += energy_increment
        
        # Update history
        self.power_history.append(total_power)
        self.last_update = current_time
    
    def get_current_metrics(self) -> EnergyMetrics:
        """Get current energy metrics"""
        self._update_energy_consumption()
        
        metrics = EnergyMetrics()
        metrics.total_energy_wh = self.total_energy_wh
        metrics.current_power_w = list(self.power_history)[-1] if self.power_history else 0.0
        metrics.budget_used_percent = (self.total_energy_wh / self.energy_budget_wh) * 100
        
        return metrics
    
    def has_energy_remaining(self, threshold_percent: float = 95.0) -> bool:
        """Check if energy budget allows continued training"""
        metrics = self.get_current_metrics()
        return metrics.budget_used_percent < threshold_percent
    
    def get_remaining_energy(self) -> float:
        """Get remaining energy budget in Wh"""
        return max(0, self.energy_budget_wh - self.total_energy_wh)
    
    def save_energy_log(self, filepath: str):
        """Save energy consumption log"""
        metrics = self.get_current_metrics()
        log_data = {
            'energy_budget_wh': self.energy_budget_wh,
            'total_energy_consumed_wh': metrics.total_energy_wh,
            'budget_utilization_percent': metrics.budget_used_percent,
            'training_duration_hours': (time.time() - self.start_time) / 3600,
            'average_power_w': sum(self.power_history) / len(self.power_history) if self.power_history else 0,
            'power_history': list(self.power_history),
            'total_energy_wh': metrics.total_energy_wh
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
    @property
    def total_energy_consumed(self):
        """Alias for total_energy_wh for API consistency"""
        return self.total_energy_wh

    @total_energy_consumed.setter
    def total_energy_consumed(self, value):
        """Allow setting total energy consumed for testing"""
        self.total_energy_wh = value

    def has_energy(self, threshold_percent: float = 95.0) -> bool:
        """Alias for has_energy_remaining for API consistency"""  
        return self.has_energy_remaining(threshold_percent)

    @property    
    def is_monitoring(self) -> bool:
        """Check if monitoring is active"""
        return self._monitoring

    def log_step(self):
        """Log current energy consumption step"""
        self._update_energy_consumption()
