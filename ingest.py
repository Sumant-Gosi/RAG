# ingest.py

from rag_backend import process_all_pdfs, split_documents, EmbeddingManager, VectorStore

PDF_DIR = "data"  # your PDF folder
PERSIST_DIR = "../data/vector_store"

if __name__ == "__main__":
    # Load PDFs
    documents = process_all_pdfs(PDF_DIR, structured=False)

    # Split into chunks
    chunks = split_documents(documents, structured=False)

    # Create embeddings
    embedder = EmbeddingManager()
    store = VectorStore(persist_dir=PERSIST_DIR)

    embeddings = embedder.embed([c.page_content for c in chunks])

    # Add to vector store
    store.add(chunks, embeddings)

    print("✅ Ingestion completed successfully!")
