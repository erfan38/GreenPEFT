import torch
import json
import os
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, List, Any

# Import your custom components
# Ensure these paths match your project structure
from ..core.energy_monitor import EnergyMonitor
from ..core.smart_sampler import EnergyAwareSampler

class GreenTrainer(Trainer):
    """
    Energy-Aware Trainer for LLaMA/PEFT.
    Replaces random sampling with 'Value-per-Watt' sampling (Loss/Length)
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
        self.energy_monitor = EnergyMonitor(energy_budget_wh)

        # 2. Initialize Smart Sampler (The "Brain")
        # We pass the dataset so it can manage indices
        self.smart_sampler = EnergyAwareSampler(
            dataset=train_dataset,
            energy_monitor=self.energy_monitor,
            base_batch_size=self.args.per_device_train_batch_size
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        OVERRIDE: Checks budget before every step.
        Updated to support new Transformers API (num_items_in_batch).
        """
        # 1. Check Energy Budget
        if not self.energy_monitor.has_energy():
            print("⚠️ Energy budget exhausted! Stopping training immediately.")
            self.control.should_training_stop = True
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # 2. Log the step's energy consumption
        self.energy_monitor.log_step()

        # 3. Proceed with standard training step (Forward + Backward)
        # We pass the new argument to the parent class to avoid the crash
        return super().training_step(model, inputs, num_items_in_batch)
    
    # def training_step(self, model, inputs):
    #     """
    #     OVERRIDE: Checks budget before every step.
    #     """
    #     # 1. Check Energy Budget
    #     if not self.energy_monitor.has_energy():
    #         print("⚠️ Energy budget exhausted! Stopping training immediately.")
    #         self.control.should_training_stop = True
    #         return torch.tensor(0.0, device=model.device, requires_grad=True)

    #     # 2. Log the step's energy consumption
    #     self.energy_monitor.log_step()

    #     # 3. Proceed with standard training step
    #     return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        OVERRIDE: Captures Per-Sample Loss and Sequence Length.
        This updates the Smart Sampler with 'Value (Loss) / Cost (Length)'.
        """
        # 1. Extract Sample Indices (CRITICAL STEP)
        # Your dataset MUST have an 'id' or 'index' column.
        if "index" in inputs:
            sample_indices = inputs.pop("index").cpu().tolist()
        elif "id" in inputs:
            sample_indices = inputs.pop("id").cpu().tolist()
        else:
            # Fallback: If no indices, just run standard loss (sampler won't update efficiently)
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        # 2. Forward Pass
        outputs = model(**inputs)
        
        # 3. Compute Per-Sample Loss (Raw, Unreduced)
        # We need to manually calculate CrossEntropy to get loss per item
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate loss per token, then average over sequence length to get loss per sample
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape back to (Batch, Seq_Len) and average rows
        loss_per_sample = loss_per_token.view(shift_labels.size(0), -1).mean(dim=1)
        
        # 4. Calculate Sequence Lengths (Energy Proxy)
        # Use attention mask to count actual non-padding tokens
        if "attention_mask" in inputs:
            # We subtract 1 because labels are shifted by 1
            lengths = (inputs["attention_mask"].sum(dim=1) - 1).cpu().tolist()
        else:
            lengths = [inputs["input_ids"].shape[1] - 1] * len(sample_indices)

        # 5. FEEDBACK LOOP: Update the Smart Sampler
        # Send (Index, Loss, Length) so it can compute Score = Loss / Length
        losses_cpu = loss_per_sample.detach().cpu().tolist()
        
        # Ensure lengths are at least 1 to avoid division errors
        lengths = [max(1, l) for l in lengths]
        
        self.smart_sampler.update_batch_outcomes(sample_indices, losses_cpu, lengths)

        # 6. Return mean loss for Backprop
        mean_loss = loss_per_sample.mean()

        return (mean_loss, outputs) if return_outputs else mean_loss

    def _save_energy_report(self, metrics):
        """
        Generates the final JSON report for carbon footprint analysis.
        """
        # Estimate CO2 (Example: 0.4 kg/kWh global avg)
        co2_emissions = (metrics.total_energy_wh / 1000.0) * 0.4

        # Safely get final loss
        final_loss = 0
        if hasattr(self, 'state') and hasattr(self.state, 'log_history') and self.state.log_history:
             final_loss = self.state.log_history[-1].get('train_loss', 0)

        report = {
            "total_energy_wh": metrics.total_energy_wh,
            "budget_used_percent": metrics.budget_used_percent,
            "training_duration_hours": (metrics.timestamp - metrics.start_time) / 3600.0 if hasattr(metrics, 'start_time') else 0,
            "co2_emissions_kg": co2_emissions,
            "final_loss": final_loss
        }

        with open("green_training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n🌱 GREEN TRAINING COMPLETED!")
        print(f"⚡ Energy used: {metrics.total_energy_wh:.2f} Wh")
        print(f"🌍 CO2 Emissions: {co2_emissions:.4f} kg")
        print(f"📄 Report saved to: green_training_report.json")