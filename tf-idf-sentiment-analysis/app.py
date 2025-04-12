import streamlit as st
import pandas as pd
import numpy as np
from embedding_utils import init_pinecone, query_pinecone, process_file_upload
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'pinecone_index' not in st.session_state:
    st.session_state.pinecone_index = init_pinecone()

# Title and description
st.title("Social Media Sentiment Analyzer")
st.markdown("""
This app analyzes sentiment in social media comments using advanced embedding technology.
* 😞 Negative (-1)
* 😐 Neutral (0)
* 😊 Positive (1)
""")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Analyze Text", "Upload Data"])

with tab1:
    # Text input for sentiment analysis
    user_input = st.text_area("Enter your comment:", height=100)
    
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            # Get prediction from Pinecone
            prediction, confidence = query_pinecone(
                user_input, 
                st.session_state.pinecone_index
            )
            
            # Map sentiment to human-readable labels
            sentiment_map = {
                -1: "Negative 😞",
                0: "Neutral 😐",
                1: "Positive 😊"
            }
            
            # Display prediction
            st.header("Prediction")
            st.markdown(f"### This comment appears to be: {sentiment_map[prediction]}")
            
            # Display confidence
            st.header("Confidence Score")
            st.progress(confidence)
            st.text(f"{confidence:.2%} confident in this prediction")

with tab2:
    st.header("Upload Training Data")
    st.markdown("""
    Upload a CSV file containing social media comments and their sentiment labels.
    
    The file should have:
    * A column containing text (with 'text' in the column name)
    * A column containing sentiment labels (with 'sentiment' or 'category' in the column name)
    * Sentiment values should be: -1 (negative), 0 (neutral), or 1 (positive)
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing and uploading data..."):
                num_samples, num_categories = process_file_upload(
                    uploaded_file,
                    st.session_state.pinecone_index
                )
                st.success(f"Successfully uploaded {num_samples} samples with {num_categories} sentiment categories!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add information section
st.markdown("---")
st.markdown("""
### About
This application uses state-of-the-art embedding technology to analyze sentiment in social media comments:

* **Embedding Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
* **Vector Database**: Pinecone for efficient similarity search
* **Sentiment Analysis**: Weighted k-NN classification based on semantic similarity

The model can be continuously improved by uploading additional labeled data through the "Upload Data" tab.
""")
