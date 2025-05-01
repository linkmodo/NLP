import streamlit as st
import os
from document_processor import DocumentProcessor
from rag_system import RAGSystem
import tempfile

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

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
        with st.spinner("Processing documents..."):
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
                response = st.session_state.rag_system.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload documents to begin.") 