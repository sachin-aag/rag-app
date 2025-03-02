import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from prepare_dataset import prepare_dataset

def train_lora():
    # Model and tokenizer initialization
    model_name = "meta-llama/Llama-2-7b-hf"  # You'll need access to this
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    
    # Prepare dataset
    dataset = prepare_dataset("cleaned_markdown_results", tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="lora-markdown-adapter",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    trainer.train()
    
    # Save trained adapter
    model.save_pretrained("lora-markdown-adapter")

if __name__ == "__main__":
    train_lora() 