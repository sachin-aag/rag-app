import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def generate_text(prompt, adapter_path, base_model="meta-llama/Llama-2-7b-hf", 
                 max_length=512, temperature=0.7, top_p=0.95):
    """
    Generate text using a LoRA fine-tuned model on Mac
    
    Args:
        prompt (str): Input prompt
        adapter_path (str): Path to the LoRA adapter weights
        base_model (str): Base model name
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling parameter
        
    Returns:
        str: Generated text
    """
    print(f"Loading base model: {base_model}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if MPS is available (for M-series Macs)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            do_sample=True
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
        
    return generated_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA fine-tuned models on Mac")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Base model name or path")
    parser.add_argument("--adapter-path", type=str, default="lora-markdown-adapter", 
                        help="Path to the LoRA adapter weights")
    parser.add_argument("--prompt", type=str, required=True, 
                        help="Input prompt for generation")
    parser.add_argument("--max-length", type=int, default=512, 
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, 
                        help="Nucleus sampling parameter")
    
    args = parser.parse_args()
    
    # Generate response
    response = generate_text(
        args.prompt,
        args.adapter_path,
        base_model=args.base_model,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print("\nGenerated Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    main() 