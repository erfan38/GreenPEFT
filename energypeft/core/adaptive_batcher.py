class EnergyAwareBatcher:
    """Dynamic batch size adaptation based on energy constraints"""
    
    def __init__(self, base_batch_size: int, energy_monitor, min_batch_size: int = 1):
        self.base_batch_size = base_batch_size
        self.energy_monitor = energy_monitor
        self.min_batch_size = min_batch_size
        self.training_progress = 0.0
        
    def get_adaptive_batch_size(self, convergence_progress: float = 0.0) -> int:
        """Calculate optimal batch size based on energy budget and progress"""
        
           # Update training progress
        self.training_progress = convergence_progress
        
        # Get energy status
        remaining_energy = self.energy_monitor.get_remaining_energy()
        
        # Handle zero or negative energy cases FIRST
        if remaining_energy <= 0:
            return self.min_batch_size  # Return minimum immediately
        
        remaining_energy_ratio = remaining_energy / self.energy_monitor.energy_budget_wh
        
        
        # Adaptive strategy based on energy and progress
        if remaining_energy_ratio > 0.7:
            # High energy: use larger batches
            energy_factor = 1.2
        elif remaining_energy_ratio > 0.3:
            # Medium energy: standard batches
            energy_factor = 1.0
        else:
            # Low energy: smaller batches
            energy_factor = 0.6
            
        # Progress factor (smaller batches later in training)
        if convergence_progress > 0.8:
            progress_factor = 0.7
        elif convergence_progress > 0.5:
            progress_factor = 0.9
        else:
            progress_factor = 1.0
            
        # Calculate final batch size
        adaptive_size = int(self.base_batch_size * energy_factor * progress_factor)
        return max(self.min_batch_size, adaptive_size)
