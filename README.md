# Retrieval-Augmented Generation (RAG) Chatbot

## Overview
This repository implements a modular Retrieval-Augmented Generation (RAG) chatbot for document-based question answering.  
The system retrieves relevant information from a vector database built over PDF documents and uses a large language model (LLM) to generate grounded responses.

Although an Electric Vehicle (EV) document set is used as a sample dataset, the architecture is **domain-agnostic**. By replacing the PDFs in the `data/` directory and rerunning the ingestion pipeline, the same system can be reused for any document collection without changing application logic.

---

## Key Features
- PDF document ingestion and chunking
- Sentence-level embeddings using `sentence-transformers`
- Persistent vector storage with ChromaDB
- Semantic retrieval with cross-encoder reranking
- Query rewriting for improved retrieval in multi-turn conversations
- Context-grounded answer generation with source citations
- REST API backend (FastAPI)
- Interactive frontend (Streamlit)

---

## System Architecture
The RAG pipeline follows a **retrieve → rerank → generate** workflow:

1. **Document Ingestion**
   - PDF documents are parsed and split into semantically meaningful chunks
   - Each chunk is embedded using a sentence embedding model
   - Embeddings and metadata are stored in a persistent vector database

2. **Query Processing**
   - User queries are optionally rewritten using recent conversation history
   - The rewritten query improves retrieval quality in multi-turn conversations

3. **Retrieval & Reranking**
   - Top-K candidate chunks are retrieved via semantic similarity search
   - A cross-encoder reranker jointly scores `(query, document)` pairs
   - Only the highest-scoring chunks are selected as context

4. **Answer Generation**
   - The LLM generates answers strictly from retrieved context
   - Source documents are attached to support grounding and traceability

---

## Project Structure

```
.
├── data/                  # Input PDF documents
├── vector_store/           # Persistent ChromaDB storage
├── ingest.py               # Document ingestion pipeline
├── rag_backend.py          # Retrieval, reranking, and RAG logic
├── rag_api.py              # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt
└── README.md
```



---

## Running the Project

### 1. Install dependencies
pip install -r requirements.txt

### 2. Set environment variables
export GROQ_API_KEY=your_api_key_here

### 3. Ingest documents
Place your PDF files inside the `data/` directory and run:
python ingest.py

### 4. Start the backend API
uvicorn rag_api:app --reload

### 5. Start the frontend UI
streamlit run streamlit_app.py


