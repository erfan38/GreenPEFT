from energypeft.core.energy_monitor import EnergyMonitor
from energypeft.core.adaptive_batcher import EnergyAwareBatcher
from energypeft.core.early_stopper import EnergyEfficiencyEarlyStopping
from energypeft.core.smart_sampler import EnergyAwareSampler
from energypeft.core.smart_sampler import GradientImportanceTracker
from energypeft.utils.carbon_scheduler import CarbonScheduler
from energypeft.integrations.huggingface_peft import HuggingFacePEFTTrainer
from energypeft.integrations.llamafactory import LlamaFactoryEnergyWrapper
from energypeft.trainers.green_trainer import GreenTrainer


class EnergyPEFT:
    """Main Energy-Aware PEFT Framework"""
    def __init__(self, energy_budget_wh: float = 100.0, base_batch_size: int = 32, importance_weight: float = 0.7):
        self.energy_monitor = EnergyMonitor(energy_budget_wh)
        self.base_batch_size = base_batch_size
        self.importance_weight = importance_weight
        
    def wrap_trainer(self, trainer_type: str = "huggingface", **kwargs):
        """Wrap existing trainer with energy awareness"""
        if trainer_type == "llamafactory":
            return LlamaFactoryEnergyWrapper(self, **kwargs)
        elif trainer_type == "huggingface":
            return HuggingFacePEFTTrainer(self, **kwargs)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

__version__ = "0.1.0"
__all__ = ["EnergyPEFT",
           "EnergyMonitor",
           "EnergyAwareBatcher",
           "EnergyEfficiencyEarlyStopping",
           "EnergyAwareSampler",
           "CarbonScheduler",
           "HuggingFacePEFTTrainer",
           "GreenTrainer",
           "GradientImportanceTracker",
           "LlamaFactoryEnergyWrapper"]
