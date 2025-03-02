import os
from datasets import Dataset
from transformers import AutoTokenizer

def load_markdown_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                # Add a prompt format to better guide the model
                content = f.read()
                formatted_text = f"Write in the author's style:\n\n{content}\n\nResponse:"
                texts.append(formatted_text)
    return texts

def prepare_dataset(directory, tokenizer, max_length=512):
    # Load all markdown files
    texts = load_markdown_files(directory)
    
    # Create dataset
    dataset = Dataset.from_dict({'text': texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset 