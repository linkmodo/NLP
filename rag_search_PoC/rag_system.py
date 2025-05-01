from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGSystem:
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        self.documents = None
        self.retriever = None
        self.llm = None
        
        # Check for API token
        self.huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.huggingface_token:
            logger.warning("HUGGINGFACE_API_TOKEN not found in environment variables")
        
    def initialize(self, documents: List[Dict]):
        """Initialize the RAG system with processed documents."""
        self.documents = documents
        
        # Create embeddings for all documents
        embeddings = []
        for doc in documents:
            embedding = self.embedding_model.encode(doc['content'])
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(embeddings)
        
        # Initialize the language model
        try:
            if self.huggingface_token:
                self.llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.7, "max_length": 512},
                    huggingfacehub_api_token=self.huggingface_token
                )
                logger.info("Language model initialized successfully")
            else:
                logger.error("Cannot initialize language model: No API token provided")
        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            self.llm = None
    
    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        if not self.vector_store or not self.documents:
            raise ValueError("RAG system not initialized. Please process documents first.")
        
        if not self.llm:
            return "Error: Language model not initialized. Please check your HuggingFace API token."
        
        # Embed the question
        question_embedding = self.embedding_model.encode(question)
        question_embedding = np.array([question_embedding]).astype('float32')
        
        # Search for similar documents
        k = 3  # Number of similar documents to retrieve
        distances, indices = self.vector_store.search(question_embedding, k)
        
        # Get the relevant documents
        relevant_docs = []
        for idx in indices[0]:
            relevant_docs.append(self.documents[idx]['content'])
        
        # Combine relevant documents
        context = "\n\n".join(relevant_docs)
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.
        If the answer cannot be found in the context, say "I don't have enough information to answer that question."

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        try:
            # Generate response
            response = self.llm(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Error: Could not generate a response. Please check your HuggingFace API token and try again." 
