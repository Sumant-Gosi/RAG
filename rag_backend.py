import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GROQ_API_KEY"] = ""  # Set to your Groq API key
  

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
# Load PDFs
# --------------------------------------------------
def process_all_pdfs(pdf_dir: str, structured: bool = False) -> List[Document]:
    documents = []

    for pdf in Path(pdf_dir).glob("**/*.pdf"):
        loader = PyMuPDFLoader(str(pdf))
        for doc in loader.load():
            doc.page_content = (
                normalize_markdown(doc.page_content)
                if structured
                else normalize_plain_text(doc.page_content)
            )
            doc.metadata = {
                "source": pdf.name,
                "page": doc.metadata.get("page", 0),
            }
            documents.append(doc)

    return documents


# --------------------------------------------------
# Split documents
# --------------------------------------------------
def split_documents(
    documents: List[Document],
    structured: bool = False,
    chunk_size: int = 700,
    chunk_overlap: int = 200,
) -> List[Document]:

    if structured:
        headers = [
            ("#", "section"),
            ("##", "subsection"),
            ("###", "subsubsection"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False,
        )
        chunks = []
        for doc in documents:
            for chunk in splitter.split_text(doc.page_content):
                chunks.append(
                    Document(page_content=chunk, metadata=doc.metadata)
                )
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


# --------------------------------------------------
# Embedding manager
# --------------------------------------------------
class EmbeddingManager:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)


# --------------------------------------------------
# Vector store
# --------------------------------------------------
class VectorStore:
    def __init__(self, persist_dir="../data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="ev_documents"
        )

    def add(self, documents: List[Document], embeddings: np.ndarray):
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in documents],
            documents=[d.page_content for d in documents],
            metadatas=[d.metadata for d in documents],
            embeddings=embeddings.tolist(),
        )


# --------------------------------------------------
# Retriever
# --------------------------------------------------
class RAGRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingManager):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed([query])[0]

        results = self.store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
        )

        docs = []
        for text, meta in zip(
            results["documents"][0], results["metadatas"][0]
        ):
            docs.append({"content": text, "metadata": meta})

        return docs


# --------------------------------------------------
# Citation helper (single source of truth)
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
# RAG with memory + fallback
# --------------------------------------------------
def rag_with_memory(
    query: str,
    retriever: RAGRetriever,
    llm: ChatGroq,
    chat_history: List[Dict[str, Any]],
    top_k: int = 5,
):
    retrieved = retriever.retrieve(query, top_k)

    # Fallback for generic chat
    if not retrieved:
        answer = llm.invoke(
            [HumanMessage(content=query)]
        ).content.strip()

        history = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer, "citations": []},
        ]
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

    answer = llm.invoke(
        [HumanMessage(content=prompt)]
    ).content.strip()

    history = chat_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer, "citations": citations},
    ]

    return answer, history, citations


# --------------------------------------------------
# Initialization
# --------------------------------------------------
PDF_DIR = "data"

documents = process_all_pdfs(PDF_DIR, structured=False)
chunks = split_documents(documents, structured=False)

embedder = EmbeddingManager()
store = VectorStore()

embeddings = embedder.embed([c.page_content for c in chunks])
store.add(chunks, embeddings)

retriever = RAGRetriever(store, embedder)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)
