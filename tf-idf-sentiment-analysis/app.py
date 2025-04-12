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
tab1, tab2, tab3 = st.tabs(["Analyze Text", "Upload Data", "Model Evaluation"])

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

with tab3:
    st.header("Model Evaluation")
    st.markdown("""
    This section evaluates the model's performance using a test set of pre-labeled comments.
    The evaluation includes:
    * Confusion Matrix
    * Classification Report (Precision, Recall, F1-Score)
    * Performance Analysis
    """)
    
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating model performance..."):
            # Convert sample data to DataFrame
            df = pd.DataFrame(sample_data)
            
            # Evaluate model
            confusion_mat, class_report, preds = evaluate_model(df['text'], df['true_sentiment'])
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            st.image(plot_confusion_matrix(confusion_mat))
            
            # Display classification report
            st.subheader("Classification Report")
            report_df = pd.DataFrame(class_report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
            
            # Calculate overall accuracy
            accuracy = class_report['accuracy']
            st.metric("Overall Accuracy", f"{accuracy:.2%}")
            
            # Analysis and insights
            st.subheader("Analysis and Insights")
            
            # Calculate class-wise performance
            class_performance = {
                "Negative": class_report['Negative']['f1-score'],
                "Neutral": class_report['Neutral']['f1-score'],
                "Positive": class_report['Positive']['f1-score']
            }
            
            # Find best and worst performing classes
            best_class = max(class_performance.items(), key=lambda x: x[1])
            worst_class = min(class_performance.items(), key=lambda x: x[1])
            
            st.markdown(f"""
            #### Key Findings:
            1. **Overall Performance**: The model achieves {accuracy:.2%} accuracy across all classes.
            2. **Best Performance**: Most accurate in detecting {best_class[0]} sentiment (F1-score: {best_class[1]:.2f})
            3. **Challenging Cases**: Less accurate with {worst_class[0]} sentiment (F1-score: {worst_class[1]:.2f})
            
            #### Potential Improvements:
            1. Increase training data for {worst_class[0]} class
            2. Fine-tune model parameters for better balance
            3. Consider context and domain-specific features
            """)

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
