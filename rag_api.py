# rag_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os

from rag_backend import rag_with_memory, get_rag_retriever
from langchain_groq import ChatGroq

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="EV-Assist RAG API", version="1.0.0", description="RAG-powered API for Electric Vehicle Q&A")

# -------------------------------
# Pydantic models
# -------------------------------
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

# -------------------------------
# Initialize retriever & LLM
# -------------------------------
retriever = get_rag_retriever(persist_dir="../data/vector_store")
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)

# -------------------------------
# Chat endpoint
# -------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    chat_history_dicts = [turn.dict() for turn in request.chat_history]

    answer, updated_history, citations = rag_with_memory(
        query=request.query,
        retriever=retriever,
        llm=llm,
        chat_history=chat_history_dicts,
    )

    history_models = [ChatTurn(**turn) for turn in updated_history]

    return ChatResponse(answer=answer, chat_history=history_models, citations=citations)

# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}
