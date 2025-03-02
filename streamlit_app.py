import streamlit as st
from rag_agent import create_rag_agent, query_rag
import re

st.set_page_config(page_title="RAG Chat Interface", layout="wide")

# Available models
AVAILABLE_MODELS = {
    "Llama 2 (7B)": "llama2:7b",
    "Llama 2 (13B)": "llama2:13b",
    "Mistral": "mistral",
    "CodeLlama": "codellama",
    "Llama 2 Uncensored": "llama2-uncensored"
}

def add_citations_to_text(text, num_sources):
    """Add citation indices at the end of sentences"""
    # Add [n] to the end of each sentence, where n is the source index
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cited_sentences = []
    
    for i, sentence in enumerate(sentences):
        if i == len(sentences) - 1 and not sentence.strip():
            continue
        # Rotate through sources if there are more sentences than sources
        source_idx = (i % num_sources) + 1
        cited_sentences.append(f"{sentence} [{source_idx}]")
    
    return " ".join(cited_sentences)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize or update query engine if model changed
    current_model = st.session_state.get("current_model", None)
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(AVAILABLE_MODELS.keys()),
        index=0
    )
    
    if "query_engine" not in st.session_state or current_model != selected_model:
        with st.spinner(f"Loading {selected_model}..."):
            st.session_state.query_engine = create_rag_agent(AVAILABLE_MODELS[selected_model])
            st.session_state.current_model = selected_model

def main():
    st.title("RAG Chat Interface")
    
    # Initialize session state and model selector
    initialize_session_state()
    
    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher values make the output more creative but less focused"
    )
    
    # Update temperature if it changed
    if hasattr(st.session_state.query_engine, 'llm'):
        st.session_state.query_engine.llm.temperature = temperature
    
    # Display current model
    st.sidebar.markdown(f"**Current Model:** {st.session_state.current_model}")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.write(message["content_with_citations"])
            else:
                st.write(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[{i}] {source['file']}**")
                        st.markdown(f"Score: {source['score']:.2f}")
                        st.markdown(f"Content: {source['content'][:200]}...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = st.session_state.query_engine.query(prompt)
            
            # Prepare sources
            sources = []
            for source_node in response.source_nodes:
                # Get metadata with fallback for url
                metadata = source_node.node.metadata
                filename = metadata.get('file_name', 'Unknown')
                url = metadata.get('url', filename)  # Use filename as fallback
                
                sources.append({
                    "file": filename,
                    "score": source_node.score,
                    "content": source_node.node.text,
                    "url": url
                })
            
            # Add citations to the response text
            response_with_citations = add_citations_to_text(
                response.response, 
                len(sources)
            )
            
            st.write(response_with_citations)
            
            # Display sources in expander
            with st.expander("View Sources"):
                for i, source in enumerate(sources, 1):
                    # Display source with or without link depending on URL availability
                    if source['url'] != source['file']:
                        st.markdown(f"**[{i}] Article: [{source['file']}](https://{source['url']})**")
                    else:
                        st.markdown(f"**[{i}] Article: {source['file']}**")
                    st.markdown(f"Score: {source['score']:.2f}")
                    st.markdown(f"Content: {source['content'][:200]}...")
                    st.markdown("---")
        
        # Save assistant message with sources and citations
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.response,
            "content_with_citations": response_with_citations,
            "sources": sources
        })

if __name__ == "__main__":
    main() 