import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Environment variables (load from .env in production)
os.getenv("TOKENIZERS_PARALLELISM")
GROQ_API_KEY = os.environ("GROQ_API_KEY")

# Text normalization for unstructured documents
def normalize_plain_text(text):
    lines = text.split("\n")
    merged = []

    Max_linelength = 80  #Chosen based on the average line length in the training doc.
    End_punc = (".", "?", "!")

    for line in lines:
        line = line.strip()

        if not line:
            merged.append("")
        elif (
            merged
            and len(merged[-1]) < Max_linelength
            and not merged[-1].endswith(End_punc)
        ):
            merged[-1] += " " + line
        else:
            merged.append(line)


    text = "\n".join(merged)
    return re.sub(r"\n{3,}", "\n\n", text)


# Text normalization for structured documents
def normalize_markdown(text):
    return re.sub(r"\n{3,}", "\n\n", text)


# PDF Processing
def process_all_pdfs(pdf_dir, structured):

    documents = [] # To store all processed documents
    for pdf_path in Path(pdf_dir).rglob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as e:
            logger.warning(f"Failed to load {pdf_path.name}: {e}")
            continue

        for page in pages:
            text = page.page_content
            if structured:
                page.page_content = normalize_markdown(text)
            else:
                page.page_content = normalize_plain_text(text)

            page.metadata.update({
                "source": pdf_path.name,
                "page": page.metadata.get("page"),
            })

        documents.append(page)

    return documents

# Split the documents into chunks
def split_documents(documents, structured, chunk_size = 500, chunk_overlap = 150):

    chunks: List[Document] = []

    if structured:
        headers = [
            ("#", "section"),
            ("##", "subsection"),
            ("###", "subsubsection"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False
        )

        for doc in documents:
            text_chunks = splitter.split_text(doc.page_content)
            for text in text_chunks:
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={**doc.metadata}  # copy metadata
                    )
                )

    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)

    return chunks


# Embedding Manager to convert the chunks into embeddings
class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed(
        self,
        texts,
        batch_size = 32
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )


# Vector Store to store the embeddings
class VectorStore:
    def __init__(self, persist_dir="../data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="ev_documents")
        logger.info(f"Connected to vector store at {persist_dir}")

    def add(self, documents, embeddings):
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in documents],
            documents=[d.page_content for d in documents],
            metadatas=[d.metadata for d in documents],
            embeddings=embeddings.tolist(),
        )


# Query Rewriter to improve query relevance
class QueryRewriter:
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
    
    def rewrite_query(self, query, chat_history):
        """
        Rewrite the query to be standalone using conversation context.
        Handles pronouns, references, and implicit context.
        """
        if not chat_history:
            return query
        
        # Only use last 3 turns for context (avoid token overflow)
        recent_history = chat_history[-6:]  
        
        history_text = "\n".join([
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
            for turn in recent_history
        ])
        
        prompt = f"""Given the conversation history, rewrite the user's latest query to be a standalone question that contains all necessary context.

Conversation History:
{history_text}

Latest Query: {query}

Rewrite the query to be clear and standalone. If the query already has full context, return it unchanged.
Only output the rewritten query, nothing else.

Rewritten Query:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query  # Fallback to original


# RAG Retriever
class RAGRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingManager, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.store = store
        self.embedder = embedder
        self.reranker = CrossEncoder(reranker_model)
        logger.info("RAG Retriever initialized with reranking")

    def retrieve(self, query, initial_k= 20, top_k = 5):
        """
        Two-stage retrieval:
        - Fast vector search for recall
        - Cross-encoder reranking for precision
        """

        try:
            query_emb = self.embedder.embed([query])[0]

            # Stage 1: Semantic search
            results = self.store.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=initial_k,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"][0]:
                logger.warning(f"No documents retrieved for query: {query}")
                return []

            # Stage 2: Rerank
            docs = []
            for text, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append({
                    "content": text,
                    "metadata": meta,
                    "distance": distance
                })

            # Reranking using cross-encoder
            reranked = self._rerank_documents(query, docs, top_k)
            
            logger.info(f"Retrieved {len(reranked)} documents for query: {query[:50]}...")
            return reranked

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _rerank_documents(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder"""
        if not docs:
            return []
        
        pairs = [(query, d["content"]) for d in docs]
        scores = self.reranker.predict(pairs)
        
        # Add rerank scores to docs
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        
        # Sort by rerank score (higher = better)
        sorted_docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_docs[:top_k]


# Citation
def extract_citations(retrieved_docs):
    citations = []
    seen = set()

    for doc in retrieved_docs:
        source, page = doc["metadata"]["source"], doc["metadata"]["page"]
        if (source, page) not in seen:
            seen.add((source, page))
            citations.append(f"{source}, page {page}")

    return citations


# RAG with Memory
def rag_with_memory(
    query,
    retriever,
    llm: ChatGroq,
    chat_history,
    query_rewriter = None,
    initial_k = 20,
    top_k = 5
):
    """
    RAG with conversation memory and query rewriting.
    
    Steps:
    1. Rewrite query using chat history for better retrieval
    2. Retrieve relevant documents
    3. Generate answer using retrieved context + conversation history
    """
    
    # Step 1: Rewrite query if we have chat history
    retrieval_query = query
    if query_rewriter and chat_history:
        retrieval_query = query_rewriter.rewrite_query(query, chat_history)
    
    # Step 2: Retrieve documents
    retrieved = retriever.retrieve(retrieval_query, initial_k=initial_k, top_k=top_k)

    # Step 3: Handle no retrieval case
    if not retrieved:
        logger.warning(f"No documents retrieved, using LLM fallback for: {query}")
        
        # Build conversation context
        conversation_context = _build_conversation_context(chat_history)
        
        fallback_prompt = f"""You are an EV assistant. No relevant documents were found in the knowledge base.

{conversation_context}

User Question: {query}

Provide a helpful response. If you don't know the answer, say so clearly and suggest what information would help."""
        
        answer = llm.invoke([HumanMessage(content=fallback_prompt)]).content.strip()
        history = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer, "citations": []}
        ]
        return answer, history, []

    # Step 4: Generate answer with context
    citations = extract_citations(retrieved)
    context = "\n\n---\n\n".join([
        f"[Source: {d['metadata']['source']}, Page {d['metadata']['page']}]\n{d['content']}"
        for d in retrieved
    ])
    
    # Build conversation context
    conversation_context = _build_conversation_context(chat_history)
    
    # Improved prompt with memory
    prompt = f"""You are an expert EV assistant. Answer the user's question using ONLY the provided context and conversation history.

{conversation_context}

Retrieved Context:
{context}

User Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain the answer, say "I don't have enough information in the knowledge base to answer this."
- Be specific and cite relevant details
- If referring to previous conversation, acknowledge it naturally
- Keep answers concise but complete

Answer:"""

    try:
        answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        logger.info(f"Generated answer with {len(citations)} citations")
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = "I apologize, but I encountered an error generating a response. Please try again."
    
    history = chat_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer, "citations": citations}
    ]
    
    return answer, history, citations


def _build_conversation_context(chat_history, max_turns = 3):
    """Build conversation context from recent history"""
    if not chat_history:
        return ""
    
    recent = chat_history[-(max_turns * 2):]  # Last N Q&A pairs
    
    context_lines = ["Previous Conversation:"]
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        context_lines.append(f"{role}: {turn['content']}")
    
    return "\n".join(context_lines)


# Load RAG Retriever 
def get_rag_retriever(persist_dir="../data/vector_store"):
    store = VectorStore(persist_dir=persist_dir)
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2",
    device="cpu",   # or "cuda" / "mps"
    normalize=True
)
    return RAGRetriever(store, embedder)

# Load Query Rewriter
def get_query_rewriter(groq_api_key: str):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=256,  
    )
    return QueryRewriter(llm)