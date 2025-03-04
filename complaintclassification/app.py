import streamlit as st
from openai import OpenAI
import pandas as pd
from pinecone import Pinecone
from typing import List
import uuid

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PINECONE_URL = "https://debt-complaints-index-judopvw.svc.aped-4627-b74a.pinecone.io"

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
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSION
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        st.stop()

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
        st.write("Upsert response:", upsert_response)
        return upsert_response
    except Exception as e:
        st.error(f"Error uploading complaint: {str(e)}")
        return None

# Function to search for similar complaints using cosine similarity
def search_similar_complaints(query_text: str, top_k: int = 5):
    try:
        query_embedding = get_embedding(query_text)
        st.write("Generated embedding dimension:", len(query_embedding))
        index = initialize_pinecone()
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default"
        )
        
        st.write("Raw query results:", results)
        return results
    except Exception as e:
        st.error(f"Error searching complaints: {str(e)}")
        return None

# Main application flow
st.title("📌Consumer Debt Complaint Analyzer")

# Sidebar: choose an option with default set to "Find Similar Complaints"
menu = st.sidebar.radio("Select an Option:", 
                          ["Find Similar Complaints", "Upload Single Complaint", "Upload CSV for Embeddings"],
                          index=0)  # Setting index=0 makes the first option default

if menu == "Upload Single Complaint":
    st.header("Upload a New Complaint")
    complaint_text = st.text_area("Enter Complaint Narrative:", height=150)
    issue_type = st.text_input("Issue Type", value="Unknown")
    
    if st.button("Upload Complaint"):
        if not complaint_text.strip():
            st.error("Please enter a complaint narrative.")
        else:
            complaint_id = str(uuid.uuid4())
            metadata = {"Issue": issue_type, "Narrative": complaint_text}
            with st.spinner("Uploading your complaint..."):
                response = upload_complaint(complaint_text, complaint_id, metadata)
                if response:
                    st.success("Complaint uploaded successfully! ID: " + complaint_id)

elif menu == "Upload CSV for Embeddings":
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

elif menu == "Find Similar Issues":
    st.header("Find Similar Issues")
    user_input = st.text_area("Enter Complaint Narrative for Search:", height=150)
    
    if st.button("Search Similar Complaints"):
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
                            
                            **Complaint Narrative:**
                            {metadata.get('Narrative', 'No narrative available')}
                            
                            ---
                            """
                        )
                else:
                    st.error("No similar complaints found in the database. Please check if the index contains data.")
                    st.write("Debug information:", results)
