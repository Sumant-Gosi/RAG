# streamlit_app_local.py

import streamlit as st
import requests
from datetime import datetime

# API Endpoint
API_URL = "http://127.0.0.1:8000/chat"
HEALTH_URL = "http://127.0.0.1:8000/health"

# Streamlit Config
st.set_page_config(
    page_title="EV-Assist",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("EV-Assist")
    st.markdown("Your Electric Vehicle Expert")
    
    # Health check
    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        if health_response.status_code == 200:
            st.success("API Connected")
        else:
            st.error("API Issues")
    except:
        st.error("API Offline")
    
    st.markdown("---")
    
    # Stats
    if "chat_history" in st.session_state:
        num_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.metric("Messages Sent", num_messages)
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Info
    with st.expander("ℹ️ About"):
        st.markdown("""
        **EV-Assist** uses RAG (Retrieval Augmented Generation) to answer questions about electric vehicles.
        
        **Features:**
        - Knowledge from EV documents
        - Conversation memory
        - Source citations
        - Powered by Llama 3.3 70B
        """)
    
    with st.expander("Example Questions"):
        st.markdown("""
        - What are the charging options?
        - Tell me about EV incentives
        """)

# Main Chat Interface
st.title("Chat with EV-Assist")

# Initialize chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display welcome message
if not st.session_state.chat_history:
    st.info("Hi! I'm your EV assistant. Ask me anything about electric vehicles!")

# Render chat history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        
        # Show citations for assistant messages
        if turn["role"] == "assistant" and turn.get("citations"):
            with st.expander(f"Sources ({len(turn['citations'])})"):
                for i, citation in enumerate(turn["citations"], start=1):
                    st.markdown(f"**[{i}]** {citation}")

# User input
query = st.chat_input("Ask about Electric Vehicles...")

if query:
    # Add user message to chat immediately
    with st.chat_message("user"):
        st.markdown(query)
    
    # Prepare payload
    payload = {
        "query": query,
        "chat_history": st.session_state.chat_history,
    }
    
    # Show loading state
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call API
                response = requests.post(API_URL, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                # Update chat history (single source of truth)
                st.session_state.chat_history = data["chat_history"]
                
                # Display answer
                st.markdown(data["answer"])
                
                # Display citations
                if data.get("citations"):
                    with st.expander(f"📚 Sources ({len(data['citations'])})"):
                        for i, citation in enumerate(data["citations"], start=1):
                            st.markdown(f"**[{i}]** {citation}")
                else:
                    st.caption("_No sources available - answer based on general knowledge_")
            
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The API might be overloaded. Please try again.")
            
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure the FastAPI server is running on http://127.0.0.1:8000")
            
            except requests.exceptions.HTTPError as e:
                st.error(f"API Error: {e.response.status_code}")
                if e.response.status_code == 500:
                    st.error("The server encountered an error. Check the API logs.")
            
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
    
    # Rerun to update the chat display
    st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by RAG • Llama 3.3 70B •")