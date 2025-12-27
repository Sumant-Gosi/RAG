"""
Ingestion script for EV-Assist RAG system.

This script is intended to be run offline whenever PDFs are added or updated.
It loads PDFs, splits them into chunks, generates embeddings, and stores them
in a persistent vector database for fast retrieval at query time.
"""

import logging
from rag_backend import (
    process_all_pdfs,
    split_documents,
    EmbeddingManager,
    VectorStore,
)

# Configuration
PDF_DIR = "data"                      # Directory containing PDFs
PERSIST_DIR = "../data/vector_store"  # Where embeddings will be stored

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting ingestion pipeline")

    try:
        # Load PDFs
        documents = process_all_pdfs(PDF_DIR, structured=False)
        logger.info(f"Loaded {len(documents)} document pages from PDFs")

        if not documents:
            logger.warning("No documents found. Exiting ingestion.")
            return

        # Split documents into chunks
        chunks = split_documents(documents, structured=False)
        logger.info(f"Split documents into {len(chunks)} chunks")

        # Initialize embedding model
        embedder = EmbeddingManager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",   # switch to "cuda" if GPU is available
            normalize=True
        )

        # Initialize vector store
        store = VectorStore(persist_dir=PERSIST_DIR)

        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        logger.info("Generating embeddings...")
        embeddings = embedder.embed(texts, batch_size=32)

        # Sanity check
        assert len(chunks) == len(embeddings), "Mismatch between chunks and embeddings"

        # Store embeddings
        store.add(chunks, embeddings)
        logger.info("Embeddings successfully added to vector store")

        logger.info("Ingestion completed successfully!")

    except Exception as e:
        logger.error("Ingestion failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
