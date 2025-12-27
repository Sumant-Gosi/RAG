# Retrieval-Augmented Generation (RAG) Chatbot

## Intro 
This repository contains a generic Retrieval-Augmented Generation (RAG) chatbot framework built around document-based question answering.
The system is designed to work with any PDF-based knowledge source. By updating the documents in the data/ directory and rerunning the ingestion pipeline, the same system can be reused for different domains without changing application logic. An Electric Vehicle (EV) document set is used as a sample use case, but the architecture itself is domain-agnostic.

## Running the project

1. Install dependencies:
pip install -r requirements.txt

2. Set environment variables:
GROQ_API_KEY=your_api_key_here

3. Ingest documents:
python ingest.py

4. Start the backend API:
uvicorn rag_api:app --reload

5. Start the frontend UI:
streamlit run streamlit_app.py
