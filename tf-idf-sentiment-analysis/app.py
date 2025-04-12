import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Social Media Comment Classifier",
    page_icon="🤖",
    layout="centered"
)

# Title and description
st.title("Social Media Comment Classifier")
st.markdown("""
This app classifies social media comments (from Reddit and Twitter) into three categories:
* -1: Negative sentiment
* 0: Neutral sentiment
* 1: Positive sentiment

The model is trained on a combined dataset of Reddit and Twitter comments to provide better coverage and accuracy across different social media platforms.
""")

def load_model():
    """Load the trained model and vectorizer"""
    if not os.path.exists('model.joblib') or not os.path.exists('vectorizer.joblib'):
        st.error("Model files not found. Please run train_model.py first to train the model.")
        st.stop()
    
    # Load the saved model and vectorizer
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for the given text"""
    # Transform the text
    text_tfidf = vectorizer.transform([text])
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    # Get prediction probabilities
    probs = model.predict_proba(text_tfidf)[0]
    return prediction, probs

# Load the model and vectorizer
model, vectorizer = load_model()

# Create text input
user_input = st.text_area("Enter your comment:", height=100)

if user_input:
    # Make prediction
    prediction, probabilities = predict_sentiment(user_input, model, vectorizer)
    
    # Map sentiment to human-readable labels
    sentiment_map = {
        -1: "Negative 😞",
        0: "Neutral 😐",
        1: "Positive 😊"
    }
    
    # Display prediction
    st.header("Prediction")
    st.markdown(f"### This comment appears to be: {sentiment_map[prediction]}")
    
    # Display confidence scores
    st.header("Confidence Scores")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Negative", f"{probabilities[0]:.2%}")
    with col2:
        st.metric("Neutral", f"{probabilities[1]:.2%}")
    with col3:
        st.metric("Positive", f"{probabilities[2]:.2%}")

# Add GitHub link and information
st.markdown("---")
st.markdown("""
### About
This app uses a machine learning model trained on a combined dataset of Reddit and Twitter comments to classify the sentiment of text. 
The model uses TF-IDF vectorization and Logistic Regression for classification.

#### Dataset Information
- Combined Reddit and Twitter comments
- Training/Testing split: 75%/25%
- Text preprocessing: TF-IDF vectorization with 5000 features
""")
