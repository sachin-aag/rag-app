# read data from cleaned_markdown_results and create a vector index using llama-index
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.text_splitter import SentenceSplitter

def convert_filename_to_url(filename: str) -> str:
    """Convert markdown filename to original URL format"""
    url = filename.split("/")[-1]
    # Remove .md extension
    url = url.replace('.md', '')
    
    # Replace www_ with www.
    url = url.replace('www_', 'www.')
    
    # Replace _com_ with .com/
    url = url.replace('_com_', '.com/')
    
    # Replace remaining underscores with hyphens
    url = url.replace('_', '-')
    
    return url

def file_metadata_fn(filename: str) -> dict:
    """Create metadata dictionary for each file"""
    url = convert_filename_to_url(filename)
    return {
        "file_name": filename,
        "url": url
    }

# Set up the embedding model
embed_model = HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')

# Read documents from the directory with metadata
documents = SimpleDirectoryReader(
    input_dir="cleaned_markdown_results",
    filename_as_id=True,
    file_metadata=file_metadata_fn
).load_data()

# Create text splitter/parser
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# Create the vector index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    text_splitter=text_splitter,
)

# Save the index to disk
index.storage_context.persist("index_storage")

