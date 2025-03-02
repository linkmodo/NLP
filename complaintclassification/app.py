import streamlit as st
from openai import OpenAI
import pandas as pd
from typing import List
from pinecone import Pinecone, ServerlessSpec

# Initialize session state for tracking initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"  # New embedding model
EMBEDDING_DIMENSION = 1536  # Dimension size for the model

# 🔹 Load API keys from Streamlit secrets and initialize clients
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
except Exception as e:
    st.error(f"Error loading API keys. Please check your .streamlit/secrets.toml file: {str(e)}")
    st.stop()

index_name = "debt-complaints-index"

# Initialize Pinecone connection
@st.cache_resource
def initialize_pinecone():
    try:
        # Get the index
        return pc.Index(index_name)
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

# Main application flow
st.title("📌 Consumer Debt Complaint Analyzer")
st.write("Enter a consumer complaint to find similar complaints from our database.")

# Initialize Pinecone connection
index = initialize_pinecone()

# User interface
user_input = st.text_area("Enter Complaint Narrative:", height=150)

if st.button("Find Similar Complaints"):
    if not user_input.strip():
        st.error("Please enter a complaint narrative.")
    else:
        try:
            with st.spinner("Analyzing your complaint..."):
                # Generate embedding for user input
                embedding = get_embedding(user_input)
                
                # Query Pinecone for similar complaints
                response = index.query(
                    vector=embedding,
                    top_k=5,  # Show 5 similar complaints
                    include_metadata=True
                )
                
                if not response["matches"]:
                    st.info("No similar complaints found in the database.")
                else:
                    st.subheader("🔗 Most Similar Complaints:")
                    for i, match in enumerate(response["matches"], 1):
                        similarity_score = match["score"] * 100  # Convert to percentage
                        st.markdown(
                            f"""
                            **Similar Complaint #{i}**
                            - **Issue Type:** {match['metadata'].get('Issue', 'Unknown')}
                            - **Similarity Score:** {similarity_score:.1f}%
                            ---
                            """
                        )

        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
