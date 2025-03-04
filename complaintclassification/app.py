import streamlit as st
from openai import OpenAI
import pandas as pd
from typing import List
from pinecone import Pinecone, ServerlessSpec

# Initialize session state for tracking initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PINECONE_URL = "https://debt-complaints-index-judopvw.svc.aped-4627-b74a.pinecone.io"

# 🔹 Load API keys from Streamlit secrets and initialize clients
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    pc = Pinecone(
        api_key=st.secrets["PINECONE_API_KEY"],
    )
except Exception as e:
    st.error(f"Error loading API keys. Please check your .streamlit/secrets.toml file: {str(e)}")
    st.stop()

# Initialize Pinecone connection
@st.cache_resource
def initialize_pinecone():
    try:
        # List available indexes
        indexes = pc.list_indexes()
        st.write("Available indexes:", [index.name for index in indexes])
        
        # Connect to the specific index using the serverless URL
        index = pc.Index(
            name="debt-complaints-index",
            host=PINECONE_URL
        )
        
        # Test the connection by getting index stats
        stats = index.describe_index_stats()
        st.write("Index stats:", stats)
        
        return index
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")
        st.stop()

# Generate embeddings with OpenAI
@st.cache_data
def get_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSION
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        st.stop()

# Function to search similar complaints
def search_similar_complaints(query_text: str, top_k: int = 5):
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(query_text)
        st.write("Generated embedding dimension:", len(query_embedding))
        
        # Query Pinecone index
        index = initialize_pinecone()
        
        # Perform the query with namespace
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default"  # Explicitly specify namespace
        )
        
        st.write("Raw query results:", results)  # Debug information
        
        return results
    except Exception as e:
        st.error(f"Error searching complaints: {str(e)}")
        return None

# Main application flow
st.title("📌 Consumer Debt Complaint Analyzer")
st.write("Enter a consumer complaint to find similar complaints from our database.")

# User interface
user_input = st.text_area("Enter Complaint Narrative:", height=150)

if st.button("Find Similar Complaints"):
    if not user_input.strip():
        st.error("Please enter a complaint narrative.")
    else:
        with st.spinner("Analyzing your complaint..."):
            results = search_similar_complaints(user_input)
            
            if results and results.get("matches"):
                st.subheader("🔗 Most Similar Complaints:")
                for i, match in enumerate(results["matches"], 1):
                    similarity_score = match["score"] * 100
                    metadata = match.get("metadata", {})
                    st.markdown(
                        f"""
                        **Similar Complaint #{i}**
                        - **Issue Type:** {metadata.get('Issue', 'Unknown')}
                        - **Similarity Score:** {similarity_score:.1f}%
                        - **ID:** {match.get('id', 'Unknown')}
                        ---
                        """
                    )
            else:
                st.error("No similar complaints found in the database. Please check if the index contains data.")
                st.write("Debug information:", results)  # Show raw results for debugging
