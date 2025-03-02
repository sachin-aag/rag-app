import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from prepare_dataset import prepare_dataset
import os
from datasets import Dataset

def load_markdown_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

def prepare_dataset(directory, tokenizer, max_length=512):
    texts = load_markdown_files(directory)
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors="pt"
        )
        # Set labels same as input_ids for causal LM training
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs
    
    # Create dataset
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_lora_optimized():
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if MPS is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model with proper device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    
    # LoRA Config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Prepare dataset with labels
    dataset = prepare_dataset("cleaned_markdown_results", tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="lora-markdown-adapter",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        use_mps_device=torch.backends.mps.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    model.save_pretrained("lora-markdown-adapter")

if __name__ == "__main__":
    train_lora_optimized() 