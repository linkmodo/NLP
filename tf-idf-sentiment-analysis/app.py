import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import io

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for model and evaluation data
if 'sentiment_analyzer' not in st.session_state:
    with st.spinner("Loading sentiment analysis model..."):
        # Check if CUDA is available, otherwise use CPU
        device = 0 if torch.cuda.is_available() else -1
        st.session_state.sentiment_analyzer = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device  # Use GPU if available, otherwise CPU
        )

# Sample test data for evaluation
sample_data = {
    'text': [
        "This is absolutely amazing!",
        "I'm not sure how I feel about this.",
        "This is terrible, I hate it.",
        "The product works as expected.",
        "Worst experience ever!",
        "Pretty good service overall.",
        "Could be better, but not bad.",
        "Outstanding performance!",
        "Complete waste of time.",
        "Just okay, nothing special."
    ],
    'true_sentiment': [1, 0, -1, 0, -1, 1, 0, 1, -1, 0]  # 1: Positive, 0: Neutral, -1: Negative
}

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

def evaluate_model(texts, true_labels):
    """Evaluate model performance on test data"""
    predictions = []
    for text in texts:
        pred, _ = analyze_sentiment(text)
        predictions.append(pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=[-1, 0, 1])
    
    # Generate classification report
    report = classification_report(true_labels, predictions, 
                                 labels=[-1, 0, 1],
                                 target_names=['Negative', 'Neutral', 'Positive'],
                                 output_dict=True)
    
    return cm, report, predictions

def plot_confusion_matrix(cm):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# Create tabs for different functionalities
tab1 = st.tabs(["Analyze Text"])

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

#### Device Information
""")

# Display device information
if torch.cuda.is_available():
    st.markdown(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    st.markdown("Running on CPU")
