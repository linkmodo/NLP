import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io
import json
import nltk

# Download the necessary NLTK tokenizers
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)
from nltk.tokenize import sent_tokenize

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

def main():
    st.title("Semantic Search Engine for Uploaded Files")
    st.write("Upload your documents below and ask questions to search for relevant content.")

    st.sidebar.header("Options")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    
    # Checkbox to choose search granularity
    sentence_search = st.checkbox("Enable sentence-level search", value=False)
    
    if uploaded_files:
        with st.spinner("Loading embedding model..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
        
        embeddings_data = []
        for file in uploaded_files:
            st.write(f"Processing file: {file.name}")
            text = extract_text(file)
            if text:
                doc_embedding = model.encode(text)
                sentences = split_text_into_sentences(text)
                if sentences:
                    sentence_embeddings = model.encode(sentences)
                else:
                    sentence_embeddings = []
                embeddings_data.append({
                    "file_name": file.name,
                    "text": text,
                    "embedding": doc_embedding.tolist(),
                    "sentences": sentences,
                    "sentence_embeddings": sentence_embeddings.tolist() if len(sentence_embeddings) else []
                })
        st.success("Files processed and embeddings computed!")
        
        if st.button("Download Embeddings JSON"):
            embeddings_json = json.dumps(embeddings_data)
            st.download_button("Download Embeddings", data=embeddings_json, file_name="embeddings.json", mime="application/json")
        
        query = st.text_input("Enter your search query")
        if query:
            query_embedding = model.encode(query)
            if sentence_search:
                sentence_results = []
                for data in embeddings_data:
                    for idx, sentence in enumerate(data["sentences"]):
                        sent_emb = np.array(data["sentence_embeddings"][idx])
                        score = cosine_similarity(query_embedding, sent_emb)
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
                    file_embedding = np.array(data["embedding"])
                    score = cosine_similarity(query_embedding, file_embedding)
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
                    st.write(f"**Extract:** {result['text'][:500]}...")
                    st.write("---")
                
                if st.button("Download Search Results as CSV"):
                    df = pd.DataFrame(results)
                    csv_data = df.to_csv(index=False)
                    st.download_button("Download CSV", data=csv_data, file_name="search_results.csv", mime="text/csv")

if __name__ == '__main__':
    main()
