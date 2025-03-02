from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
import torch

def create_rag_agent(model_name: str = "llama2:7b") -> RetrieverQueryEngine:
    """
    Create a RAG query engine with the specified model
    
    Args:
        model_name (str): Name of the Ollama model to use
        
    Returns:
        RetrieverQueryEngine: Configured query engine
    """
    # Ensure CUDA is available if you have a GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up the embedding model with explicit device
    embed_model = HuggingFaceEmbedding(
        model_name='intfloat/multilingual-e5-large',
        device=device,
        embed_batch_size=1
    )
    
    # Create Ollama LLM instance with selected model
    llm = Ollama(model=model_name, temperature=0.1)
    
    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load the index with the specified settings
    storage_context = StorageContext.from_defaults(persist_dir="index_storage")
    index = load_index_from_storage(storage_context)
    
    # Create retriever with similarity threshold
    # configure retriever
    retriever = index.as_retriever(
        similarity_top_k=10,
        diversity_top_k=5, # Ensure that the selected nodes are diverse.
    )
    
    # Add reranking step
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5
    )
    
    # Create response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact"
    )
    
    # Create query engine with reranker
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[rerank],
        response_synthesizer=response_synthesizer,
    )
    
    return query_engine

def query_rag(query_engine, question: str, streaming_callback=None):
    """
    Query the RAG system and get response with citations
    """
    # Enable streaming if callback is provided
    response = query_engine.query(
        question,
        streaming=True if streaming_callback else False,
        streaming_callback=streaming_callback
    )
    
    # Return both response and sources for non-streaming case
    return {
        "answer": response.response,
        "sources": [
            {
                "file_name": node.node.metadata['file_name'],
                "score": node.score,
                "content": node.node.text[:200]
            }
            for node in response.source_nodes
        ]
    }

if __name__ == "__main__":
    # Example usage
    query_engine = create_rag_agent()
    
    # Example question
    question = "What is wrong with American economy?"
    query_rag(query_engine, question) 