import torch
import json
import os
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Optional

# Import your custom components
from ..core.energy_monitor import EnergyMonitor
from ..core.smart_sampler import EnergyAwareSampler

class GreenTrainer(Trainer):
    """
    Energy-Aware Trainer for LLaMA.
    Replaces random sampling with 'Value-per-Watt' sampling
    and enforces a strict energy budget.
    """

    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None,
                 energy_budget_wh=100.0, importance_weight=0.7, **kwargs):

        # Initialize parent Trainer
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            **kwargs
        )

        self.energy_budget_wh = energy_budget_wh

        # 1. Initialize Energy Monitoring
        # This tracks GPU/CPU usage in the background
        self.energy_monitor = EnergyMonitor(energy_budget_wh)

        # 2. Initialize Smart Sampler (The "Brain")
        # We pass the dataset so it can pre-calculate sequence lengths (Energy Cost)
        self.smart_sampler = EnergyAwareSampler(
            dataset=train_dataset,
            energy_monitor=self.energy_monitor,
            base_batch_size=self.args.per_device_train_batch_size,
            importance_weight=importance_weight
        )

        print(f"🌱 GreenTrainer initialized with {energy_budget_wh}Wh energy budget")

    def get_train_dataloader(self) -> DataLoader:
        """
        OVERRIDE: Force the Trainer to use our EnergyAwareSampler
        instead of the default RandomSampler.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.smart_sampler,  # <--- Injecting our smart logic here
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """
        OVERRIDE: Wraps the training loop with energy monitoring.
        """
        # Start background monitoring thread
        self.energy_monitor.start_monitoring()
        print("⚡ Starting energy-aware training...")

        try:
            # Run standard Hugging Face training
            result = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial, 
                **kwargs
            )
            return result

        finally:
            # Always stop monitoring and save report, even if training crashes
            metrics = self.energy_monitor.stop_monitoring()
            self._save_energy_report(metrics)

    def training_step(self, model, inputs):
        """
        OVERRIDE: Checks budget before every step.
        """
        # 1. Check Energy Budget
        if not self.energy_monitor.has_energy():
            print("⚠️ Energy budget exhausted! Stopping training immediately.")
            self.control.should_training_stop = True
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # 2. Log the step's energy consumption
        self.energy_monitor.log_step()

        # 3. Proceed with standard training step (Forward + Backward)
        # Note: We don't calculate gradients manually anymore. 
        # We let 'compute_loss' handle the feedback.
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        OVERRIDE: Captures Loss to update the Smart Sampler.
        This solves the 'Chicken and Egg' problem by using Loss instead of Gradients.
        """
        # 1. Run the Forward Pass (Standard)
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # 2. FEEDBACK LOOP: Send loss back to Sampler
        # We use .item() to detach from graph and avoid memory leaks
        current_loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # The sampler needs to know the 'value' of the batch we just processed.
        # High Loss = High Learning Opportunity for next time.
        # We update the sampler's internal scores.
        self.smart_sampler.update_batch_outcomes(current_loss_value)

        return (loss, outputs) if return_outputs else loss

    def _save_energy_report(self, metrics):
        """
        Generates the final JSON report for carbon footprint analysis.
        """
        # Estimate CO2 (Example: 0.4 kg/kWh global avg)
        co2_emissions = (metrics.total_energy_wh / 1000.0) * 0.4

        report = {
            "total_energy_wh": metrics.total_energy_wh,
            "budget_used_percent": metrics.budget_used_percent,
            "training_duration_hours": (metrics.timestamp - metrics.start_time) / 3600.0 if hasattr(metrics, 'start_time') else 0,
            "co2_emissions_kg": co2_emissions,
            "final_loss": getattr(self.state, 'log_history', [{}])[-1].get('train_loss', 0)
        }

        with open("green_training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n🌱 GREEN TRAINING COMPLETED!")
        print(f"⚡ Energy used: {metrics.total_energy_wh:.2f} Wh")
        print(f"🌍 CO2 Emissions: {co2_emissions:.4f} kg")
        print(f"📄 Report saved to: green_training_report.json")
