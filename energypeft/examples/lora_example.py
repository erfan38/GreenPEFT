import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from energypeft import EnergyPEFT
from energypeft.core.energy_monitor import EnergyMonitor
from energypeft.integrations.huggingface_peft import HuggingFacePEFTTrainer


def main():
    # Initialize energy framework
    energy_peft = EnergyPEFT(energy_budget_wh=30.0)
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Load dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
    
    # Wrap with energy-aware trainer
    trainer = energy_peft.wrap_trainer(
        trainer_type="huggingface",
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset
    )
    
    # Train with energy awareness
    trainer.train()

if __name__ == "__main__":
    main()
