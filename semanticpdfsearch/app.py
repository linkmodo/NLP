import streamlit as st
import numpy as np
import pandas as pd
import PyPDF2
import docx
import json
from openai import OpenAI
import nltk

# Set the page configuration with the new title.
st.set_page_config(page_title="Semantic Text Embedding Generation And Search Tool")

# Set up OpenAI client with API key from Streamlit secrets.
client = OpenAI(
  api_key=st.secrets["OPENAI_API_KEY"],
)

# Download only the necessary NLTK resource for sentence tokenization.
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.warning("Could not download required NLTK resources. Please check your configuration.")

from nltk.tokenize import sent_tokenize

# Inject custom CSS for a white background, a background image, and legible text.
st.markdown(
    """
    <style>
    .stApp {
        background: white;
        background-image: url('https://raw.githubusercontent.com/linkmodo/NLP/refs/heads/main/semanticpdfsearch/white_bg.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Make text color dark for better visibility on white background */
    .stMarkdown, .stText, p, h1, h2, h3 {
        color: #333333 !important;
    }
    
    /* Style buttons */
    .stButton>button {
        background-color: #f0f0f0;
        color: #333333;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    
    /* Style text inputs */
    .stTextInput>div>div>input {
        color: #333333;
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    /* Style file uploader */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* Style sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px);
    }

    /* Style sidebar content */
    .css-1d391kg > div {
        background-color: transparent !important;
    }
    
    /* Style sidebar title */
    .css-1d391kg .css-17lntkn {
        color: #333333 !important;
    }
    
    /* Style sidebar text */
    [data-testid="stSidebar"] .stMarkdown {
        color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def extract_text(file):
    """Extract text from different file types."""
    file_type = file.name.split('.')[-1].lower()
    text = ""
    if file_type == "pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file_type in ["doc", "docx"]:
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_type == "txt":
        text = file.read().decode("utf-8")
    elif file_type == "csv":
        df = pd.read_csv(file)
        text = df.to_csv(index=False)
    else:
        st.warning(f"Unsupported file type: {file_type}")
    return text

def split_text_into_sentences(text):
    """Break down long text into a list of sentences."""
    return sent_tokenize(text)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text):
    """
    Generate an embedding for the provided text using OpenAI's API.
    Using the text-embedding-3-small model.
    """
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def main():
    st.title("Semantic Text Embedding Generation And Search Tool")
    st.write("Upload your documents, generate embeddings, and search for relevant content.")

    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OPENAI_API_KEY not found in Streamlit secrets. Please add it to your secrets.toml file.")
        return

    st.sidebar.header("Options")
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    
    # Checkbox to choose between document-level and sentence-level search.
    sentence_search = st.sidebar.checkbox("Enable sentence-level search", value=False)
    
    # In-memory dataset for embeddings.
    embeddings_data = []
    if uploaded_files:
        with st.spinner("Processing files and generating embeddings..."):
            for file in uploaded_files:
                st.write(f"Processing file: {file.name}")
                text = extract_text(file)
                if text:
                    doc_embedding = get_embedding(text)
                    sentences = split_text_into_sentences(text)
                    if sentences:
                        sentence_embeddings = get_embedding(" ".join(sentences))
                    else:
                        sentence_embeddings = []
                    embeddings_data.append({
                        "file_name": file.name,
                        "text": text,
                        "embedding": doc_embedding,
                        "sentences": sentences,
                        "sentence_embeddings": sentence_embeddings
                    })
        st.success("Files processed and embeddings computed!")
        
        if st.button("Download Embeddings Dataset JSON"):
            embeddings_json = json.dumps(embeddings_data)
            st.download_button("Download Embeddings", data=embeddings_json, file_name="embeddings.json", mime="application/json")

    query = st.text_input("Enter your search query")
    if query and embeddings_data:
        with st.spinner("Generating query embedding and searching..."):
            query_embedding = get_embedding(query)
            if sentence_search:
                sentence_results = []
                for data in embeddings_data:
                    for sentence in data["sentences"]:
                        sent_emb = get_embedding(sentence)
                        score = cosine_similarity(query_embedding, np.array(sent_emb))
                        sentence_results.append({
                            "file_name": data["file_name"],
                            "sentence": sentence,
                            "score": score
                        })
                sentence_results = sorted(sentence_results, key=lambda x: x["score"], reverse=True)
                
                st.write("### Sentence-Level Search Results")
                for res in sentence_results:
                    st.write(f"**File:** {res['file_name']}")
                    st.write(f"**Score:** {res['score']:.4f}")
                    st.write(f"**Sentence:** {res['sentence']}")
                    st.write("---")
                
                if st.button("Download Sentence Search Results as CSV"):
                    df = pd.DataFrame(sentence_results)
                    csv_data = df.to_csv(index=False)
                    st.download_button("Download CSV", data=csv_data, file_name="sentence_search_results.csv", mime="text/csv")
            else:
                results = []
                for data in embeddings_data:
                    score = cosine_similarity(query_embedding, np.array(data["embedding"]))
                    results.append({
                        "file_name": data["file_name"],
                        "score": score,
                        "text": data["text"]
                    })
                results = sorted(results, key=lambda x: x["score"], reverse=True)
                
                st.write("### Document-Level Search Results")
                for result in results:
                    st.write(f"**File:** {result['file_name']}")
                    st.write(f"**Score:** {result['score']:.4f}")
                    st.write(f"**Snippet:** {result['text'][:500]}...")
                    st.write("---")
                
                if st.button("Download Search Results as CSV"):
                    df = pd.DataFrame(results)
                    csv_data = df.to_csv(index=False)
                    st.download_button("Download CSV", data=csv_data, file_name="search_results.csv", mime="text/csv")

if __name__ == '__main__':
    main()
