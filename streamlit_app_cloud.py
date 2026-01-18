# streamlit_app.py

import streamlit as st
import os
from datetime import datetime

from rag_backend import (
    rag_with_memory,
    get_rag_retriever,
    get_query_rewriter,
)
from langchain_groq import ChatGroq

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Environment & initialization
# --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please configure it in environment variables or Streamlit secrets.")
    st.stop()

@st.cache_resource
def initialize_rag():
    """Initialize RAG components once per session"""
    retriever = get_rag_retriever(persist_dir="data/vector_store")
    query_rewriter = get_query_rewriter(GROQ_API_KEY)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )
    return retriever, query_rewriter, llm


retriever, query_rewriter, llm = initialize_rag()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📚 RAG Chatbot")
    st.markdown(
        "A generic Retrieval-Augmented Generation chatbot. "
        "Upload domain PDFs, re-ingest, and ask questions grounded in your documents."
    )

    st.success("RAG pipeline initialized")

    st.markdown("---")

    if "chat_history" in st.session_state:
        user_turns = [m for m in st.session_state.chat_history if m["role"] == "user"]
        st.metric("Questions Asked", len(user_turns))

    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    with st.expander("ℹ️ About"):
        st.markdown(
            """
            **How it works**
            - Documents are embedded and stored in a vector database
            - User queries retrieve relevant chunks
            - Retrieved context is passed to the LLM for grounded answers

            This demo runs the full RAG pipeline directly inside Streamlit.
            """
        )

# --------------------------------------------------
# Main chat interface
# --------------------------------------------------
st.title("Chat with your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.info("Ask a question about the documents you ingested.")

# Render chat history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("citations"):
            with st.expander(f"Sources ({len(turn['citations'])})"):
                for i, citation in enumerate(turn["citations"], start=1):
                    st.markdown(f"**[{i}]** {citation}")

# User input
query = st.chat_input("Ask a question...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, history, citations = rag_with_memory(
                query=query,
                retriever=retriever,
                llm=llm,
                chat_history=st.session_state.chat_history,
                query_rewriter=query_rewriter,
            )

            st.session_state.chat_history = history
            st.markdown(answer)

            if citations:
                with st.expander(f"📄 Sources ({len(citations)})"):
                    for i, citation in enumerate(citations, start=1):
                        st.markdown(f"**[{i}]** {citation}")
            else:
                st.caption("_No sources available for this response_")

    st.rerun()

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Generic RAG Demo • Streamlit + Vector Search + LLM")
