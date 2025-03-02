# 🦙 Personal RAG Assistant

An intelligent RAG (Retrieval-Augmented Generation) system built with LlamaIndex that provides insights from your favorite articles and content.

## 🌟 Features

- **Smart Retrieval**: Uses advanced embedding and reranking to find the most relevant content
- **Multiple LLM Support**: Switch between different Ollama models (Llama2, Mistral, etc.)
- **Interactive UI**: Clean Streamlit interface with adjustable parameters
- **Source Citations**: Every response includes citations and links to original articles
- **URL Metadata**: Maintains original article URLs for easy reference

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- Required models pulled in Ollama:
  ```bash
  ollama pull llama2:7b
  ollama pull mistral
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Build the index:
   ```bash
   python index_data_llamaind.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🛠️ Architecture 