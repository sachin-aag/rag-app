import streamlit as st
from rag_agent import create_rag_agent, query_rag

def initialize_session():
    """Initialize session state and RAG agent"""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = create_rag_agent()

def main():
    st.title("Prof G's RAG Assistant")
    st.subheader("Ask questions about Scott Galloway's articles")
    
    # Initialize RAG
    initialize_session()
    
    # Add temperature slider in sidebar
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher values make the output more creative but less focused"
    )
    
    # Update temperature if it changed
    if hasattr(st.session_state.rag_engine, 'llm'):
        st.session_state.rag_engine.llm.temperature = temperature
    
    # Create a container for the answer and sources
    response_container = st.container()
    
    # Add the query input at the bottom
    with st.container():
        st.write("---")  # Add a separator
        query = st.text_input("What would you like to know?", key="query_input")
    
    if query:
        # Get response from RAG
        with st.spinner("Searching and generating response..."):
            result = query_rag(st.session_state.rag_engine, query)
            
            with response_container:
                # Display answer with citations
                st.write("### Answer")
                st.write(result["answer"])
                
                # Display sources with URLs
                st.write("### Sources")
                for idx, source in enumerate(result["sources"], 1):
                    with st.expander(f"[{idx}] Source: {source['file_name']}"):
                        st.write(f"Relevance Score: {source['score']:.2f}")
                        
                        # Display URL if available in metadata
                        if 'url' in source:
                            st.write(f"URL: [https://{source['url']}](https://{source['url']})")
                        
                        st.write(f"Preview: {source['content']}...")

if __name__ == "__main__":
    main() 