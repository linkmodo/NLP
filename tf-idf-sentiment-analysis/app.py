import streamlit as st
from transformers import pipeline
import torch

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for model
if 'sentiment_analyzer' not in st.session_state:
    with st.spinner("Loading sentiment analysis model..."):
        # Check if CUDA is available, otherwise use CPU
        device = 0 if torch.cuda.is_available() else -1
        st.session_state.sentiment_analyzer = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device  # Use GPU if available, otherwise CPU
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
        'NEGATIVE': -1,
        'NEUTRAL': 0,
        'POSITIVE': 1
    }
    
    return label_map[label], score

# Create text input
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
