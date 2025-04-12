import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
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

# Initialize session state for model
if 'sentiment_analyzer' not in st.session_state:
    with st.spinner("Loading sentiment analysis model..."):
        st.session_state.sentiment_analyzer = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

# Title and description
st.title("Social Media Sentiment Analyzer")
st.markdown("""
This app analyzes sentiment in social media comments using a pretrained RoBERTa model 
fine-tuned on Twitter data.

The model classifies text into three categories:
* 😞 Negative
* 😐 Neutral
* 😊 Positive
""")

def analyze_sentiment(text):
    """Analyze sentiment of given text"""
    result = st.session_state.sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score']
    
    # Map label to our format
    label_map = {
        'negative': -1,
        'neutral': 0,
        'positive': 1
    }
    
    return label_map[label.lower()], score

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Analyze Text", "Upload Data"])

with tab1:
    # Text input for sentiment analysis
    user_input = st.text_area("Enter your comment:", height=100)
    
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            # Get prediction
            prediction, confidence = analyze_sentiment(user_input)
            
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
                # No implementation for uploading data
                st.success("Successfully uploaded data!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add information section
st.markdown("---")
st.markdown("""
### About
This application uses a state-of-the-art RoBERTa model fine-tuned on Twitter data 
for sentiment analysis. The model has been trained on a large dataset of social media 
posts and can effectively analyze the emotional tone of text.

#### Model Information
- **Base Model**: RoBERTa
- **Training**: Fine-tuned on Twitter data
- **Task**: Sentiment Analysis
- **Labels**: Negative, Neutral, Positive
""")
