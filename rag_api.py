from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
import logging

from rag_backend import (
    rag_with_memory,
    get_rag_retriever,
    get_query_rewriter
)
from langchain_groq import ChatGroq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="EV-Assist RAG API",
    version="2.0.0",
    description= "Backend API for a RAG chatbot. "
        "Handles retrieval, query rewriting, and conversation memory."
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatTurn(BaseModel):
    role: str
    content: str
    citations: List[str] = []

class ChatRequest(BaseModel):
    query: str
    chat_history: List[ChatTurn] = []

class ChatResponse(BaseModel):
    answer: str
    chat_history: List[ChatTurn]
    citations: List[str]

# Initialization
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

retriever = get_rag_retriever(persist_dir="../vector_store")
query_rewriter = get_query_rewriter(GROQ_API_KEY)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)

logger.info("RAG API initialized successfully")

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint.

    We rewrite the user's query using recent chat history to improve retrieval,
    then pass both retrieved context and conversation history to the LLM.
    This keeps answers relevant in multi-turn conversations.
    """
    try:
        chat_history_dicts = [turn.dict() for turn in request.chat_history]

        logger.info(f"Processing query: {request.query}")
        logger.info(f"Chat history length: {len(chat_history_dicts)}")

        answer, updated_history, citations = rag_with_memory(
            query=request.query,
            retriever=retriever,
            llm=llm,
            chat_history=chat_history_dicts,
            query_rewriter=query_rewriter,
            initial_k=20,
            top_k=5
        )

        history_models = [ChatTurn(**turn) for turn in updated_history]

        return ChatResponse(
            answer=answer,
            chat_history=history_models,
            citations=citations
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "retriever": "initialized",
        "llm": "initialized",
        "query_rewriter": "initialized"
    }

# Debug endpoint
@app.post("/debug/retrieve")
def debug_retrieve(query: str, top_k: int = 5):
    """Debug endpoint to test retrieval without LLM generation"""
    try:
        retrieved = retriever.retrieve(query, initial_k=20, top_k=top_k)
        return {
            "query": query,
            "retrieved_count": len(retrieved),
            "documents": [
                {
                    "content": doc["content"][:200] + "...",
                    "metadata": doc["metadata"],
                    "rerank_score": doc.get("rerank_score", 0)
                }
                for doc in retrieved
            ]
        }
    except Exception as e:
        logger.error(f"Debug retrieve error: {e}")
        raise HTTPException(status_code=500,detail="Internal server error")