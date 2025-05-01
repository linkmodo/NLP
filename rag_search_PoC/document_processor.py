import os
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """Process all documents in a directory and return processed chunks."""
        documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_docs = self._process_file(file_path)
                documents.extend(file_docs)
        
        return documents
    
    def _process_file(self, file_path: str) -> List[Dict]:
        """Process a single file based on its extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load and split the document
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        
        # Convert to dictionary format
        processed_chunks = []
        for chunk in chunks:
            processed_chunks.append({
                'content': chunk.page_content,
                'metadata': chunk.metadata
            })
        
        return processed_chunks 