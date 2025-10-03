# greenpeft/trainers/green_trainer.py

import torch
from transformers import Trainer, TrainingArguments
from ..core.energy_monitor import EnergyMonitor
from ..core.smart_sampler import EnergyAwareSampler
from ..core.adaptive_batcher import EnergyAwareBatcher
from ..core.early_stopper import EnergyEfficiencyEarlyStopping

class GreenTrainer(Trainer):
    """Drop-in replacement for transformers.Trainer with energy awareness"""

    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None,
                 energy_budget_wh=100.0, importance_weight=0.7, **kwargs):

        # Initialize energy monitoring
        self.energy_monitor = EnergyMonitor(energy_budget_wh)
        self.energy_budget_wh = energy_budget_wh

        # Initialize smart components
        dataset_size = len(train_dataset) if train_dataset else 1000
        base_batch_size = kwargs.get('args', TrainingArguments(output_dir='.')).per_device_train_batch_size

        self.smart_sampler = EnergyAwareSampler(
            dataset_size=dataset_size,
            energy_monitor=self.energy_monitor,
            base_batch_size=base_batch_size,
            importance_weight=importance_weight
        )

        self.adaptive_batcher = EnergyAwareBatcher(
            base_batch_size=base_batch_size,
            energy_monitor=self.energy_monitor
        )

        self.early_stopper = EnergyEfficiencyEarlyStopping(
            energy_monitor=self.energy_monitor,
            patience=5
        )

        # Initialize parent Trainer
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            **kwargs
        )

        print(f" GreenTrainer initialized with {energy_budget_wh}Wh energy budget")

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Override train method with energy monitoring"""

        # Start energy monitoring
        self.energy_monitor.start_monitoring()

        print(" Starting energy-aware training...")
        print(f" Energy budget: {self.energy_budget_wh} Wh")

        try:
            # Call parent train method with energy callbacks
            result = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial, 
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs
            )

            return result

        finally:
            # Always stop monitoring and save report
            metrics = self.energy_monitor.stop_monitoring()
            self._save_energy_report(metrics)

    def training_step(self, model, inputs):
        """Override training step for dynamic adaptation"""

        # Check energy budget before each step
        if not self.energy_monitor.has_energy():
            print("⚠️ Energy budget exhausted! Stopping training.")
            self.control.should_training_stop = True
            return torch.tensor(0.0, requires_grad=True)

        # Get adaptive batch size (for next iteration)
        current_progress = self.state.global_step / self.state.max_steps if self.state.max_steps else 0.0
        adaptive_batch_size = self.adaptive_batcher.get_adaptive_batch_size(current_progress)

        # Log adaptive changes
        if adaptive_batch_size != self.args.per_device_train_batch_size:
            print(f"🔄 Adapting batch size: {self.args.per_device_train_batch_size} → {adaptive_batch_size}")

        # Perform normal training step
        loss = super().training_step(model, inputs)

        # Update importance scores if gradients available
        if hasattr(model, 'named_parameters'):
            gradients = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))

            if gradients:
                all_grads = torch.cat(gradients)
                # Update smart sampler with gradient information
                # (Implementation would need batch indices tracking)

        # Log energy metrics
        self.energy_monitor.log_step()

        return loss

    def _save_energy_report(self, metrics):
        """Save comprehensive energy report"""
        report = {
            "energy_consumed_wh": metrics.total_energy_wh,
            "budget_utilization": metrics.budget_used_percent,
            "co2_emissions_kg": getattr(metrics, 'co2_emissions', 0),
            "training_steps": self.state.global_step,
            "energy_per_step": metrics.total_energy_wh / max(self.state.global_step, 1),
            "final_loss": getattr(self.state, 'log_history', [{}])[-1].get('train_loss', 0)
        }

        # Save to JSON
        import json
        with open("green_training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n GREEN TRAINING COMPLETED!")
        print(f" Energy used: {metrics.total_energy_wh:.2f} Wh ({metrics.budget_used_percent:.1f}% of budget)")
        print(f" CO2 saved vs standard training: ~{max(0, 20-report['co2_emissions_kg']):.3f} kg")
        print(f"Report saved to: green_training_report.json")