import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import json
from sklearn.neighbors import NearestNeighbors

def extract_text(file):
    """Extract text from various file types."""
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
        # For CSV files, read into a DataFrame and convert to CSV string format.
        df = pd.read_csv(file)
        text = df.to_csv(index=False)
    else:
        st.warning(f"Unsupported file type: {file_type}")
    return text

def main():
    st.title("Semantic Search Engine for Uploaded Files (k-NN)")
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
        
        # Search functionality using k-NN
        query = st.text_input("Enter your search query")
        if query:
            query_embedding = model.encode(query)
            embeddings_matrix = np.array([data["embedding"] for data in embeddings_data])
            
            # Allow the user to choose the number of nearest neighbors (k)
            k = st.slider("Select number of nearest neighbors (k)", min_value=1, 
                          max_value=min(len(embeddings_data), 10), value=3)
            
            # Build and query the k-NN model using cosine distance
            knn = NearestNeighbors(n_neighbors=k, metric="cosine")
            knn.fit(embeddings_matrix)
            distances, indices = knn.kneighbors([query_embedding])
            
            st.write("### Search Results using k‑Nearest Neighbors")
            for i, idx in enumerate(indices[0]):
                result = embeddings_data[idx]
                # Convert cosine distance to similarity score: similarity = 1 - distance
                similarity = 1 - distances[0][i]
                st.write(f"**File:** {result['file_name']}")
                st.write(f"**Similarity Score:** {similarity:.4f}")
                st.write(f"**Extract:** {result['text'][:500]}...")
                st.write("---")
            
            # Option to save search results as CSV
            if st.button("Download Search Results as CSV"):
                results_list = []
                for i, idx in enumerate(indices[0]):
                    result = embeddings_data[idx]
                    similarity = 1 - distances[0][i]
                    results_list.append({
                        "file_name": result["file_name"],
                        "similarity_score": similarity,
                        "text_extract": result["text"][:500]
                    })
                df = pd.DataFrame(results_list)
                csv_data = df.to_csv(index=False)
                st.download_button("Download CSV", data=csv_data, file_name="search_results.csv", mime="text/csv")

if __name__ == '__main__':
    main()
