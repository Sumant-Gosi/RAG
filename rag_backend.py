# rag_backend.py

import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --------------------------------------------------
# Text normalization
# --------------------------------------------------
def normalize_plain_text(text: str) -> str:
    lines = text.split("\n")
    merged = []

    for line in lines:
        line = line.strip()
        if not line:
            merged.append("")
        elif merged and len(merged[-1]) < 80:
            merged[-1] += " " + line
        else:
            merged.append(line)

    text = "\n".join(merged)
    return re.sub(r"\n{3,}", "\n\n", text)


def normalize_markdown(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


# --------------------------------------------------
# PDF Processing Helpers
# --------------------------------------------------
def process_all_pdfs(pdf_dir: str, structured: bool = False) -> List[Document]:
    documents = []

    for pdf in Path(pdf_dir).glob("**/*.pdf"):
        loader = PyMuPDFLoader(str(pdf))
        for doc in loader.load():
            doc.page_content = normalize_markdown(doc.page_content) if structured else normalize_plain_text(doc.page_content)
            doc.metadata = {"source": pdf.name, "page": doc.metadata.get("page", 0)}
            documents.append(doc)

    return documents


def split_documents(documents: List[Document], structured: bool = False, chunk_size: int = 500, chunk_overlap: int = 150) -> List[Document]:
    if structured:
        headers = [("#", "section"), ("##", "subsection"), ("###", "subsubsection")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
        chunks = []
        for doc in documents:
            for chunk in splitter.split_text(doc.page_content):
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        return chunks

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


# --------------------------------------------------
# Embedding Manager
# --------------------------------------------------
class EmbeddingManager:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)


# --------------------------------------------------
# Vector Store
# --------------------------------------------------
class VectorStore:
    def __init__(self, persist_dir="../data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="ev_documents")

    def add(self, documents: List[Document], embeddings: np.ndarray):
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in documents],
            documents=[d.page_content for d in documents],
            metadatas=[d.metadata for d in documents],
            embeddings=embeddings.tolist(),
        )


# --------------------------------------------------
# RAG Retriever
# --------------------------------------------------
class RAGRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingManager, similarity_threshold=0.9):
        self.store = store
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, top_k: int = 5):
        query_emb = self.embedder.embed([query])[0]

        results = self.store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma uses distance (lower = more similar)
            if distance <= self.similarity_threshold:
                docs.append({"content": text, "metadata": meta, "distance": distance})
            
            print(distance)

        return docs


# --------------------------------------------------
# Citation Helper
# --------------------------------------------------
def extract_citations(retrieved_docs: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    citations = []

    for d in retrieved_docs:
        key = (d["metadata"]["source"], d["metadata"]["page"])
        if key not in seen:
            seen.add(key)
            citations.append(f"{key[0]}, page {key[1]}")
    return citations


# --------------------------------------------------
# RAG with Memory + Fallback
# --------------------------------------------------
def rag_with_memory(query: str, retriever: RAGRetriever, llm: ChatGroq, chat_history: List[Dict[str, Any]], top_k: int = 10):
    retrieved = retriever.retrieve(query, 10)
    retrieved = rerank_documents(query, retrieved, top_k=5)

    if not retrieved:
        answer = llm.invoke([HumanMessage(content=query)]).content.strip()
        history = chat_history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer, "citations": []}]
        return answer, history, []

    citations = extract_citations(retrieved)
    context = "\n\n".join(d["content"] for d in retrieved)

    prompt = f"""
Answer ONLY using the provided context.
Do NOT hallucinate.

Context:
{context}

Question:
{query}
"""

    answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    history = chat_history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer, "citations": citations}]
    return answer, history, citations


# --------------------------------------------------
# Utility: Load RAG Retriever from pre-built store
# --------------------------------------------------
def get_rag_retriever(persist_dir="../data/vector_store"):
    store = VectorStore(persist_dir=persist_dir)
    embedder = EmbeddingManager()
    return RAGRetriever(store, embedder)

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, docs: List[Dict[str, Any]], top_k: int = 5):
    pairs = [(query, d["content"]) for d in docs]
    scores = reranker_model.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs[:top_k]