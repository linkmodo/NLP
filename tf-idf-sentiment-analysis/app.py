import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        model_path = "https://raw.githubusercontent.com/linkmodo/NLP/main/tf-idf-sentiment-analysis/models/model.joblib"
        vectorizer_path = "https://raw.githubusercontent.com/linkmodo/NLP/main/tf-idf-sentiment-analysis/models/vectorizer.joblib"
        
        if not model_path.exists() or not vectorizer_path.exists():
            st.error("Model files not found. Please ensure the model files are in the 'models' folder.")
            return None, None
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, model, vectorizer):
    try:
        # Transform text using the vectorizer
        text_features = vectorizer.transform([text])
        
        # Get prediction and probability
        prediction = model.predict(text_features)[0]
        probabilities = model.predict_proba(text_features)[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def main():
    st.title("Sentiment Analysis App")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
        .sentiment-positive {
            color: green;
            font-size: 24px;
        }
        .sentiment-negative {
            color: red;
            font-size: 24px;
        }
        .confidence-bar {
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load model and vectorizer
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Input text area
    st.subheader("Enter your text for sentiment analysis")
    text = st.text_area("", height=150, placeholder="Type your text here...")
    
    if text:
        # Get prediction
        prediction, confidence = predict_sentiment(text, model, vectorizer)
        
        if prediction is not None:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment")
                if prediction == 1:
                    st.markdown('<p class="sentiment-positive">Positive 😊</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="sentiment-negative">Negative 😞</p>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("Confidence")
                # Create confidence bar
                confidence_percent = confidence * 100
                st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {'#4CAF50' if prediction == 1 else '#f44336'};"></div>
                    </div>
                    <p style="text-align: center;">{confidence_percent:.1f}%</p>
                """, unsafe_allow_html=True)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: grey; font-size: 14px;">
            Built with Streamlit • Powered by TF-IDF
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
