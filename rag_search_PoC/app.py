import streamlit as st
import os
from document_processor import DocumentProcessor
from rag_system import RAGSystem
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

# Check for API token
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not huggingface_token:
    st.error("HUGGINGFACE_API_TOKEN not found in environment variables. Please add it to your .env file.")

# Page config
st.set_page_config(
    page_title="RAG Search Utility",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 RAG Search Utility")
st.markdown("""
This application allows you to upload documents (PDF, Word, or text files) and ask questions about their content.
The system will retrieve relevant information using a RAG (Retrieval-Augmented Generation) approach.
""")

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_processed:
        if not huggingface_token:
            st.error("Please add your HuggingFace API token to the .env file before uploading documents.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Create temporary directory for uploaded files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded files
                        for file in uploaded_files:
                            file_path = os.path.join(temp_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                        
                        # Initialize document processor
                        processor = DocumentProcessor()
                        documents = processor.process_directory(temp_dir)
                        
                        # Initialize RAG system
                        st.session_state.rag_system = RAGSystem()
                        st.session_state.rag_system.initialize(documents)
                        st.session_state.documents_processed = True
                        
                    st.success("Documents processed successfully!")
                except Exception as e:
                    logger.error(f"Error processing documents: {str(e)}")
                    st.error(f"Error processing documents: {str(e)}")

# Main chat interface
if st.session_state.documents_processed:
    st.header("Ask Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_system.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    error_message = "An error occurred while generating the response. Please try again."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Please upload documents to begin.") 
