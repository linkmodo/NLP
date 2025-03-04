import streamlit as st
from openai import OpenAI
import pandas as pd
from pinecone import Pinecone
from typing import List, Dict, Tuple
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PINECONE_URL = "https://debt-complaints-index-judopvw.svc.aped-4627-b74a.pinecone.io"

# Predefined categories with their descriptions
COMPLAINT_CATEGORIES = {
    "Harassment": "Aggressive collection practices, repeated calls, threats, or inappropriate contact",
    "Billing Dispute": "Incorrect amounts, unauthorized charges, or fee-related issues",
    "Identity Theft": "Unauthorized accounts, fraudulent activity, or identity verification issues",
    "Credit Reporting": "Inaccurate credit reporting, disputes with credit bureaus, or reporting errors",
    "Payment Processing": "Issues with payments not being processed, applied incorrectly, or delayed",
    "Communication Issues": "Lack of response, unclear information, or language barriers",
    "Legal Concerns": "Issues related to lawsuits, legal notices, or statutory violations",
    "Account Management": "Problems with account status, balance, or servicing",
    "Settlement Issues": "Problems with debt settlement, negotiations, or resolution attempts",
    "Documentation": "Missing or incorrect documentation, verification requests"
}

# Category embeddings cache
category_embeddings = {}

# 🔹 Load API keys from Streamlit secrets and initialize clients
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
except Exception as e:
    st.error(f"Error loading API keys. Please check your .streamlit/secrets.toml file: {str(e)}")
    st.stop()

# Initialize Pinecone connection
@st.cache_resource
def initialize_pinecone():
    try:
        # Connect to the specific index using the serverless URL
        index = pc.Index(
            name="debt-complaints-index",
            host=PINECONE_URL
        )
        return index
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")
        st.stop()

# Generate embeddings with OpenAI
@st.cache_data
def get_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSION
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        st.stop()

@st.cache_data
def get_category_embedding(category: str, description: str) -> List[float]:
    """Generate embedding for a category using its name and description"""
    try:
        text = f"{category}: {description}"
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSION
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating category embedding: {str(e)}")
        return None

def initialize_category_embeddings():
    """Initialize embeddings for all categories"""
    for category, description in COMPLAINT_CATEGORIES.items():
        if category not in category_embeddings:
            category_embeddings[category] = get_category_embedding(category, description)

def classify_complaint(complaint_embedding: List[float]) -> Tuple[str, float]:
    """Classify a complaint based on its embedding"""
    initialize_category_embeddings()
    
    # Calculate similarity with each category
    similarities = {}
    for category, category_emb in category_embeddings.items():
        similarity = cosine_similarity(
            np.array(complaint_embedding).reshape(1, -1),
            np.array(category_emb).reshape(1, -1)
        )[0][0]
        similarities[category] = similarity
    
    # Get the most similar category
    best_category = max(similarities.items(), key=lambda x: x[1])
    return best_category[0], best_category[1]

# Function to upload a single complaint to Pinecone
def upload_complaint(complaint_text: str, complaint_id: str, metadata: dict):
    try:
        embedding = get_embedding(complaint_text)
        index = initialize_pinecone()
        
        vector_item = {
            "id": complaint_id,
            "values": embedding,
            "metadata": metadata
        }
        
        upsert_response = index.upsert(vectors=[vector_item], namespace="default")
        return upsert_response
    except Exception as e:
        st.error(f"Error uploading complaint: {str(e)}")
        return None

# Function to search for similar complaints using cosine similarity
def search_similar_complaints(query_text: str, top_k: int = 5):
    try:
        query_embedding = get_embedding(query_text)
        index = initialize_pinecone()
        
        # Classify the query
        category, confidence = classify_complaint(query_embedding)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default"
        )
        
        return results, category, confidence
    except Exception as e:
        st.error(f"Error searching complaints: {str(e)}")
        return None, None, None

# Main application flow
st.title("📌 Consumer Debt Complaint Analyzer 📌")

# Sidebar: choose an option with default set to "Find Similar Complaints"
menu = st.sidebar.radio("Select an Option:", 
                          ["Find Similar Complaints", "Upload CSV for Embeddings"],
                          index=0)  # Setting index=0 makes the first option default

if menu == "Upload CSV for Embeddings":
    st.header("Upload CSV File for Embedding Generation")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("CSV Data Preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
        
        if not df.empty:
            # Use 'narrative' column by default
            text_column = 'narrative'
            issue_column = st.selectbox("Select column for issue type (optional):", ["None"] + list(df.columns))
            
            if st.button("Process CSV"):
                with st.spinner("Generating embeddings and uploading to Pinecone..."):
                    index = initialize_pinecone()
                    for i, row in df.iterrows():
                        complaint_text = str(row[text_column])
                        # Use selected issue column if provided
                        if issue_column != "None":
                            metadata = {
                                "Issue": row[issue_column],
                                "Narrative": complaint_text
                            }
                        else:
                            metadata = {
                                "Issue": "Unknown",
                                "Narrative": complaint_text
                            }
                        complaint_id = str(uuid.uuid4())
                        embedding = get_embedding(complaint_text)
                        vector_item = {
                            "id": complaint_id,
                            "values": embedding,
                            "metadata": metadata
                        }
                        # Upsert the vector for each row
                        index.upsert(vectors=[vector_item], namespace="default")
                    st.success("CSV data processed and embeddings uploaded successfully!")

elif menu == "Find Similar Complaints":
    st.header("Find Similar Complaints")
    user_input = st.text_area("Enter Complaint Narrative for Search:", height=150)
    
    if st.button("Search Similar Complaints"):
        if not user_input.strip():
            st.error("Please enter a complaint narrative.")
        else:
            with st.spinner("Analyzing your complaint..."):
                results, category, confidence = search_similar_complaints(user_input)
                
                if category:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**🎯 Category:** {category}")
                        st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
                    with col2:
                        st.markdown(f"**Description:** {COMPLAINT_CATEGORIES[category]}")
                    st.markdown("---")
                
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
                            
                            **Complaint Narrative:**
                            {metadata.get('Narrative', 'No narrative available')}
                            
                            ---
                            """
                        )
                else:
                    st.error("No similar complaints found in the database.")

# Add footer at the very end of the file
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
    Built by Li Fan, 2025 - Powered by Streamlit, Pinecone and OpenAI.
    </div>
    """,
    unsafe_allow_html=True
)
