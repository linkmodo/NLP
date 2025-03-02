import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time
import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# 🔹 Load API keys from Streamlit secrets
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
except Exception as e:
    st.error(f"Error loading API keys. Please check your .streamlit/secrets.toml file: {str(e)}")
    st.stop()

index_name = "debt-complaints-index"

# Ensure the index exists, else create it
@st.cache_resource
def initialize_pinecone():
    try:
        # List all indexes
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]

        # Create index if it doesn't exist
        if index_name not in index_names:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # Get the index
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        st.stop()

index = initialize_pinecone()

# Load consumer complaints dataset with error handling
@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        if not os.path.exists("consumer_complaints.csv"):
            st.error("Dataset file not found. Please ensure 'consumer_complaints.csv' exists in the project directory.")
            st.stop()

        df = pd.read_csv("/content/consumer_complaints.csv")
        df = df[df['Product'] == 'Debt collection']
        df = df[df['Consumer complaint narrative'].notna()]

        if len(df) == 0:
            st.warning("No valid complaints found in the dataset.")
            st.stop()

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

# Generate embeddings with rate limiting and error handling
@st.cache_data
def get_embedding(text: str) -> List[float]:
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        st.stop()

@st.cache_data
def generate_embeddings(df: pd.DataFrame):
    with st.spinner("Generating embeddings... This may take a while."):
        vectors_to_upsert = []
        for i, row in df.iterrows():
            try:
                embedding = get_embedding(row["Consumer complaint narrative"])
                meta = {"Issue": row["Issue"]}
                vectors_to_upsert.append((str(row["complaint_id"]), embedding, meta))
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                st.warning(f"Error processing complaint {row['complaint_id']}: {str(e)}")
                continue

        # Batch upload to Pinecone with error handling
        try:
            for i in range(0, len(vectors_to_upsert), 100):
                batch = vectors_to_upsert[i:i+100]
                index.upsert(
                    vectors=[(id_, vec, meta) for id_, vec, meta in batch]
                )
        except Exception as e:
            st.error(f"Error uploading to Pinecone: {str(e)}")
            st.stop()

# Train classifier with progress indication
@st.cache_resource
def train_classifier(df: pd.DataFrame):
    try:
        with st.spinner("Training classifier..."):
            X = np.array([get_embedding(text) for text in df["Consumer complaint narrative"]])
            y = df["encoded_issue"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)

            # Calculate and display accuracy
            accuracy = clf.score(X_test, y_test)
            st.info(f"Model accuracy: {accuracy:.2f}")

            return clf
    except Exception as e:
        st.error(f"Error training classifier: {str(e)}")
        st.stop()

# Main application flow
st.title("Consumer Debt Complaint Analyzer")
st.write("Enter a consumer complaint to predict its issue category and find similar complaints.")

# Load data and initialize models
df = load_data()
label_encoder = LabelEncoder()
df["encoded_issue"] = label_encoder.fit_transform(df["Issue"])

if not st.session_state.model_trained:
    generate_embeddings(df)
    clf = train_classifier(df)
    st.session_state.model_trained = True
    st.session_state.classifier = clf

# User interface
user_input = st.text_area("Enter Complaint Narrative:", height=150)

if st.button("Analyze Complaint"):
    if not user_input.strip():
        st.error("Please enter a complaint narrative.")
    else:
        try:
            # Predict issue category
            embedding = get_embedding(user_input)
            label_id = st.session_state.classifier.predict(np.array(embedding).reshape(1, -1))[0]
            predicted_issue = label_encoder.inverse_transform([label_id])[0]

            st.subheader(f"🔍 Predicted Issue Category: **{predicted_issue}**")

            # Find similar complaints
            st.subheader("🔗 Top Similar Complaints:")
            response = index.query(vector=embedding, top_k=3, include_metadata=True)

            if not response["matches"]:
                st.info("No similar complaints found.")
            else:
                for match in response["matches"]:
                    st.write(
                        f"- (ID {match['id']}) **Issue:** {match['metadata'].get('Issue', 'Unknown')} "
                        f"– **Similarity:** {match['score']:.2f}"
                    )
        except Exception as e:
            st.error(f"Error analyzing complaint: {str(e)}")

# Modified generate_embeddings function
def generate_embeddings(df: pd.DataFrame):
    """Generates and uploads embeddings to Pinecone, handling potential Streamlit context issues."""
    with st.spinner("Generating embeddings... This may take a while."):
        vectors_to_upsert = []
        for i, row in df.iterrows():
            try:
                embedding = get_embedding(row["Consumer complaint narrative"])
                meta = {"Issue": row["Issue"]}
                vectors_to_upsert.append((str(row["complaint_id"]), embedding, meta))
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                # Accumulate warnings/errors instead of immediately displaying them
                st.session_state.setdefault('embedding_errors', []).append(f"Error processing complaint {row['complaint_id']}: {str(e)}")
                continue

        # Batch upload to Pinecone with error handling
        try:
            for i in range(0, len(vectors_to_upsert), 100):
                batch = vectors_to_upsert[i:i+100]
                index.upsert(
                    vectors=[(id_, vec, meta) for id_, vec, meta in batch]
                )
        except Exception as e:
            st.session_state['embedding_errors'] = [f"Error uploading to Pinecone: {str(e)}"]  # Store the error

    # Display accumulated warnings/errors outside the cached function
    if 'embedding_errors' in st.session_state:
        for error in st.session_state['embedding_errors']:
            st.warning(error)  # Now display the accumulated warnings
        del st.session_state['embedding_errors']  # Clear errors after displaying

# Main application flow (modified to handle the classifier training outside the cached function)
st.title("Consumer Debt Complaint Analyzer")
st.write("Enter a consumer complaint to predict its issue category and find similar complaints.")

# Load data
df = load_data()
label_encoder = LabelEncoder()
df["encoded_issue"] = label_encoder.fit_transform(df["Issue"])


# Initialize Pinecone index (if not already initialized)
index = initialize_pinecone()

# Train the classifier *outside* the cached function
if not st.session_state.model_trained:
    with st.spinner("Training classifier and generating embeddings..."):
        generate_embeddings(df)  # Generate and upload embeddings

        try:  # Put classifier training in a try-except block
            X = np.array([get_embedding(text) for text in df["Consumer complaint narrative"]])
            y = df["encoded_issue"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            st.info(f"Model accuracy: {accuracy:.2f}")
            st.session_state.classifier = clf
            st.session_state.model_trained = True  # Set the flag after successful training

        except Exception as e:
            st.error(f"Error during classifier training: {str(e)}")
            st.stop()

# Add a loading indicator while processing the complaint
if st.button("Analyze Complaint"):
    if not user_input.strip():
        st.error("Please enter a complaint narrative.")
    else:
        with st.spinner("Analyzing complaint..."): # Improved UI feedback
          # Existing code to analyze the complaint
          try:
              # ... (your existing prediction and similarity code)
          except Exception as e:
              st.error(f"Error analyzing complaint: {str(e)}")
