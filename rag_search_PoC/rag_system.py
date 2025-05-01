from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from langchain.chains import RetrievalQA
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, huggingface_token: str = None):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        self.documents = None
        self.retriever = None
        self.chat_client = None
        self.huggingface_token = huggingface_token
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        
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
        
        # Initialize the chat client using Together.ai
        together_key = self.together_api_key
        if not together_key:
            raise ValueError("Environment variable TOGETHER_API_KEY not set.")
        openai.api_key = together_key
        openai.api_base = "https://api.together.xyz/v1"
        self.chat_client = openai.OpenAI(
            api_key=together_key,
            base_url="https://api.together.xyz/v1"
        )
        # Test chat client
        test_resp = self.chat_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        )
        logger.info("Together.ai chat client initialized successfully")
    
    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        if not self.vector_store or not self.documents:
            raise ValueError("RAG system not initialized. Please process documents first.")
        
        if not hasattr(self, 'chat_client') or not self.chat_client:
            return "Error: Chat client not initialized. Check TOGETHER_API_KEY and logs."
        
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
        
        try:
            # Generate response with Together.ai chat
            messages = [
                {"role": "system", "content": f"Based on the following context, please answer the question. If the answer cannot be found in the context, say \"I don't have enough information to answer that question.\"\n\nContext:\n{context}"},
                {"role": "user", "content": question}
            ]
            resp = self.chat_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=messages
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: Could not generate a response. Details: {str(e)}"
