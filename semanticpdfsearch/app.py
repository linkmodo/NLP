import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io
import json

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
        # For CSV files, we read the file into a DataFrame and convert it to CSV string format.
        df = pd.read_csv(file)
        text = df.to_csv(index=False)
    else:
        st.warning(f"Unsupported file type: {file_type}")
    return text

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    st.title("Semantic Search Engine for Uploaded Files")
    st.write("Upload your documents below and ask questions to search for relevant content.")

    # Sidebar options
    st.sidebar.header("Options")
    
    # File uploader (accepting multiple files)
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Load the embedding model
        with st.spinner("Loading embedding model..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Process uploaded files and compute embeddings
        embeddings_data = []
        for file in uploaded_files:
            st.write(f"Processing file: {file.name}")
            text = extract_text(file)
            if text:
                # Compute embedding for the entire text
                embedding = model.encode(text)
                embeddings_data.append({
                    "file_name": file.name,
                    "text": text,
                    "embedding": embedding.tolist()  # Convert to list for JSON serialization
                })
        st.success("Files processed and embeddings computed!")
        
        # Option to save embeddings for future use
        if st.button("Download Embeddings JSON"):
            embeddings_json = json.dumps(embeddings_data)
            st.download_button("Download Embeddings", data=embeddings_json, file_name="embeddings.json", mime="application/json")
        
        # Search functionality
        query = st.text_input("Enter your search query")
        if query:
            query_embedding = model.encode(query)
            results = []
            for data in embeddings_data:
                file_embedding = np.array(data["embedding"])
                score = cosine_similarity(query_embedding, file_embedding)
                results.append({
                    "file_name": data["file_name"],
                    "score": score,
                    "text": data["text"]
                })
            
            # Sort results by descending similarity score
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            st.write("### Search Results")
            for result in results:
                st.write(f"**File:** {result['file_name']}")
                st.write(f"**Score:** {result['score']:.4f}")
                # Display the first 500 characters of the extracted text as a snippet
                st.write(f"**Extract:** {result['text'][:500]}...")
                st.write("---")
            
            # Option to save search results as CSV
            if st.button("Download Search Results as CSV"):
                df = pd.DataFrame(results)
                csv_data = df.to_csv(index=False)
                st.download_button("Download CSV", data=csv_data, file_name="search_results.csv", mime="text/csv")

if __name__ == '__main__':
    main()
