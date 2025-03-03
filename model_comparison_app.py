import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag_agent import create_rag_agent, query_rag
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import time

# Page configuration
st.set_page_config(
    page_title="LLM Model Comparison",
    page_icon="🔍",
    layout="wide"
)

# Cache the models to avoid reloading
@st.cache_resource
def load_vanilla_model():
    """Load vanilla Llama model via Ollama"""
    return Ollama(model="llama2:7b")

@st.cache_resource
def load_rag_model():
    """Load RAG-enhanced model"""
    return create_rag_agent(model_name="llama2:7b")

@st.cache_resource
def load_lora_model():
    """Load LoRA fine-tuned model"""
    base_model = "meta-llama/Llama-2-7b-hf"
    adapter_path = "lora-markdown-adapter"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if MPS is available (for M-series Macs)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_vanilla_response(prompt, temperature=0.7):
    """Generate response from vanilla Llama model"""
    llm = load_vanilla_model()
    llm.temperature = temperature
    return llm.complete(prompt)

def generate_rag_response(prompt, temperature=0.7):
    """Generate response from RAG-enhanced model"""
    # Update the global LLM temperature setting
    Settings.llm.temperature = temperature
    
    # Get the RAG engine
    rag_engine = load_rag_model()
    
    # Generate response
    result = query_rag(rag_engine, prompt)
    return result

def generate_lora_response(prompt, temperature=0.7, max_length=512):
    """Generate response from LoRA fine-tuned model"""
    model, tokenizer = load_lora_model()
    
    # Get device
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
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
    st.title("🔍 LLM Model Comparison")
    st.markdown("""
    Compare responses from three different models:
    - **Vanilla Llama 7B**: Base model without enhancements
    - **RAG-enhanced Llama 7B**: Retrieval-Augmented Generation using Prof G's articles
    - **LoRA fine-tuned Llama 7B**: Model fine-tuned on Prof G's writing style
    """)
    
    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Higher values make output more creative but less focused"
    )
    
    # Input prompt
    prompt = st.text_area("Enter your prompt:", height=100)
    
    # Generate button
    if st.button("Generate Responses"):
        if not prompt:
            st.warning("Please enter a prompt.")
        else:
            # Create three columns for results
            col1, col2, col3 = st.columns(3)
            
            with st.spinner("Generating responses..."):
                # Track timing
                start_time = time.time()
                
                # Generate responses in parallel using st.cache
                with col1:
                    st.subheader("Vanilla Llama 7B")
                    vanilla_start = time.time()
                    vanilla_response = generate_vanilla_response(prompt, temperature)
                    vanilla_time = time.time() - vanilla_start
                    st.write(vanilla_response)
                    st.caption(f"Generation time: {vanilla_time:.2f} seconds")
                
                with col2:
                    st.subheader("RAG-enhanced Llama 7B")
                    rag_start = time.time()
                    rag_result = generate_rag_response(prompt, temperature)
                    rag_time = time.time() - rag_start
                    st.write(rag_result["answer"])
                    
                    # Display sources
                    with st.expander("View Sources"):
                        for idx, source in enumerate(rag_result["sources"], 1):
                            st.markdown(f"**Source {idx}:** {source['file_name']}")
                            if 'url' in source:
                                st.markdown(f"[Link to article](https://{source['url']})")
                            st.markdown(f"Relevance: {source['score']:.2f}")
                            st.markdown(f"Preview: {source['content']}...")
                            st.markdown("---")
                    
                    st.caption(f"Generation time: {rag_time:.2f} seconds")
                
                with col3:
                    st.subheader("LoRA Fine-tuned Llama 7B")
                    lora_start = time.time()
                    lora_response = generate_lora_response(prompt, temperature)
                    lora_time = time.time() - lora_start
                    st.write(lora_response)
                    st.caption(f"Generation time: {lora_time:.2f} seconds")
                
                total_time = time.time() - start_time
                st.success(f"All responses generated in {total_time:.2f} seconds")
            
            # User feedback section
            st.subheader("Which response did you prefer?")
            feedback = st.radio(
                "Select the best response:",
                ["Vanilla Llama 7B", "RAG-enhanced Llama 7B", "LoRA Fine-tuned Llama 7B", "None of them"]
            )
            
            if st.button("Submit Feedback"):
                st.balloons()
                st.success(f"Thank you for your feedback! You preferred: {feedback}")
                # Here you could log the feedback to a file or database

if __name__ == "__main__":
    main() 