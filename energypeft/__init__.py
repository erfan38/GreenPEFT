from energypeft.core.energy_monitor import EnergyMonitor
from energypeft.core.adaptive_batcher import EnergyAwareBatcher  
from energypeft.core.early_stopper import EnergyEfficiencyEarlyStopping
# CHANGE: Import the new classes here
from energypeft.core.smart_sampler import EnergyAwareSampler, LossEfficiencyTracker
from energypeft.trainers.green_trainer import GreenTrainer

# Main entry point class
class EnergyPEFT:
    def __init__(self, energy_budget_wh=100.0, base_batch_size=32, importance_weight=0.7):
        self.energy_budget_wh = energy_budget_wh
        self.base_batch_size = base_batch_size
        self.importance_weight = importance_weight
        self.monitor = EnergyMonitor(energy_budget_wh)
        
    def wrap_trainer(self, trainer_type="huggingface", **kwargs):
        """
        Wraps a standard trainer with GreenPEFT capabilities.
        """
        if trainer_type == "huggingface":
            return GreenTrainer(
                energy_budget_wh=self.energy_budget_wh,
                importance_weight=self.importance_weight,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

__all__ = [
    "EnergyPEFT", 
    "EnergyMonitor", 
    "EnergyAwareBatcher", 
    "EnergyEfficiencyEarlyStopping", 
    "EnergyAwareSampler", 
    "LossEfficiencyTracker",
    "GreenTrainer"
]