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
from docling.document_converter import DocumentConverter

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Environment variables (load from .env in production)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Text normalization for unstructured documents
def normalize_plain_text(text):
    lines = text.split("\n")
    merged = []

    Max_linelength = 80  # Based on an average line length in the training data
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

    documents = [] 
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

# # PDF Processing using Docling
# def process_all_pdfs_docling(pdf_dir):
#     converter = DocumentConverter()
#     documents = []
    
#     for pdf_path in Path(pdf_dir).rglob("*.pdf"):
#         result = converter.convert(str(pdf_path))
#         markdown = result.document.export_to_markdown()
        
#         documents.append(Document(
#             page_content=markdown,
#             metadata={"source": pdf_path.name}
#         ))
    
#     return documents


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
                        page_content=text.page_content,
                        metadata={**doc.metadata, **text.metadata}  
                    )
                )            
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)

    return chunks


# Converting the chunks into embeddings
class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed(self,texts,batch_size = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )


# Storing the embeddings
class VectorStore:
    def __init__(self, persist_dir="../data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="ev_documents")
        logger.info(f"Connected to vector store at {persist_dir}")

    def reset(self):
        self.client.delete_collection("ev_documents")
        self.collection = self.client.get_or_create_collection("ev_documents")


    def add(self, documents, embeddings):
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in documents],
            documents=[d.page_content for d in documents],
            metadatas=[d.metadata for d in documents],
            embeddings=embeddings.tolist(),
        )


# Improving query relevance
class QueryRewriter:
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
    
    def rewrite_query(self, query, chat_history):
        """
        Rewrite the query to be standalone using conversation context.
        To particularly take care of pronouns, references, and implicit context.
        """
        if not chat_history:
            return query
        
        # Only use last 3 turns for context to avoid token overflow
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
        """Retrieve and then rerank documents for the given query"""
        try:
            query_emb = self.embedder.embed([query])[0]

            # Semantic search
            results = self.store.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=initial_k,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"][0]:
                logger.warning(f"No documents retrieved for query: {query}")
                return []

            # Rerank
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
        """Rerank documents using a cross-encoder"""
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
        print(source, page)
        if (source, page) not in seen:
            seen.add((source, page))
            citations.append(f"{source}, page {page}")

    return citations


# RAG 
def rag_with_memory(
    query: str,
    retriever,
    llm,
    chat_history: list,
    query_rewriter=None,
    initial_k=20,
    top_k=5
):

    # Query rewriting 
    retrieval_query = query
    if query_rewriter and chat_history:
        retrieval_query = query_rewriter.rewrite_query(query, chat_history)

    # Intent detection
    intent_prompt = f"""
Classify the user query.

- Greetings, small talk, or meta questions → GENERAL
- Questions requiring document knowledge → RAG

Respond with ONLY one word.

Query: "{query}"
"""
    try:
        intent = llm.invoke([HumanMessage(content=intent_prompt)]).content.strip().upper()
    except Exception:
        intent = "RAG"

    # General chat -> no retrieval
    if intent == "GENERAL":
        general_prompt = f"""
You are a friendly EV assistant.
Respond naturally. Do NOT cite sources.

Conversation History:
{_build_conversation_context(chat_history)}

User Question: {query}
"""
        answer = llm.invoke([HumanMessage(content=general_prompt)]).content.strip()

        updated_history = chat_history + [
            {"role": "user", "content": query, "citations": []},
            {"role": "assistant", "content": answer, "citations": []},
        ]
        return answer, updated_history, []

    # Retrieve documents
    try:
        retrieved_docs = retriever.retrieve(
            retrieval_query,
            initial_k=initial_k,
            top_k=top_k,
        )
    except Exception:
        retrieved_docs = []

    # Relevance filtering
    RELEVANCE_LENGTH_THRESHOLD = 80
    relevant_docs = [
        d for d in retrieved_docs
        if d.get("content") and len(d["content"].strip()) >= RELEVANCE_LENGTH_THRESHOLD
    ]

    # If no context found, then fallback
    if not relevant_docs:
        answer = "I don't have enough information in the knowledge base to answer this."
        updated_history = chat_history + [
            {"role": "user", "content": query, "citations": []},
            {"role": "assistant", "content": answer, "citations": []},
        ]
        return answer, updated_history, []

    # Generate context
    context = build_context(relevant_docs, max_tokens=1200)
    print("Context for the query: ", context)


    rag_prompt = f"""
You are an expert EV assistant.

Use ONLY the provided context to answer the question.

Conversation History:
{_build_conversation_context(chat_history)}

Retrieved Context:
{context}

User Question:
{query}

Rules:
- Answer strictly from the context
- If the answer is not present, say:
  "I don't have enough information in the knowledge base to answer this."

Answer:
"""
    answer = llm.invoke([HumanMessage(content=rag_prompt)]).content.strip()

    # Grounding verification
    grounded = is_answer_grounded(llm, query, answer, context)

    # Provide citations if grounded only    
    citations = extract_citations(relevant_docs) if grounded else []

    # Update memory
    updated_history = chat_history + [
        {"role": "user", "content": query, "citations": []},
        {"role": "assistant", "content": answer, "citations": citations},
    ]

    return answer, updated_history, citations


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

def is_answer_grounded(llm, question: str, answer: str, context: str) -> bool:
    """
    Uses the LLM to verify whether the answer is strictly supported
    by the retrieved context.
    """

    prompt = f"""
You are verifying answer grounding in a RAG system.

Question:
{question}

Answer:
{answer}

Retrieved Context:
{context}

Is the answer FULLY supported by the retrieved context?

Respond with ONLY one word:
GROUNDED or NOT_GROUNDED
"""
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        return result == "GROUNDED"
    except Exception:
        return False
    
def build_context(
    docs,
    max_tokens=1200,
    approx_tokens_per_char=0.25
):
    seen_contents = set()
    context_blocks = []
    total_tokens = 0

    for d in docs:
        content = d["content"].strip()

        # Remove identical chunks
        if content in seen_contents:
            continue
        seen_contents.add(content)

        # Inline metadata
        source = d["metadata"].get("source", "unknown")
        page = d["metadata"].get("page", "N/A")

        block = (
            f"[Source: {source} | Page: {page}]\n"
            f"{content}"
        )

        # Token budget check
        block_tokens = int(len(block) * approx_tokens_per_char)
        if total_tokens + block_tokens > max_tokens:
            break

        context_blocks.append(block)
        total_tokens += block_tokens

    return "\n\n".join(context_blocks)