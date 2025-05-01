# RAG Search Utility

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about their content. The system uses semantic search to find relevant information and generates answers using a language model.

## Features

- Support for multiple document types (PDF, Word, Text)
- Efficient document processing and chunking
- Semantic search using FAISS
- Chatbot interface using Streamlit
- Open-source models and tools

## Prerequisites

- Python 3.8 or higher
- HuggingFace API token

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-model-poc
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your HuggingFace API token:
```
HUGGINGFACE_API_TOKEN=your_token_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your documents using the sidebar

4. Ask questions about the content of your documents in the chat interface

## How It Works

1. **Document Processing**: Documents are loaded and split into chunks using LangChain's document loaders and text splitters.

2. **Embedding**: Each document chunk is converted into a vector embedding using the SentenceTransformer model.

3. **Indexing**: The embeddings are stored in a FAISS index for efficient similarity search.

4. **Querying**: When a question is asked:
   - The question is converted into an embedding
   - Similar document chunks are retrieved using FAISS
   - The relevant context is passed to the language model
   - The model generates an answer based on the context

## Technologies Used

- Streamlit for the web interface
- LangChain for document processing
- SentenceTransformers for embeddings
- FAISS for vector similarity search
- HuggingFace Hub for the language model

## License

This project is open source and available under the MIT License. 
