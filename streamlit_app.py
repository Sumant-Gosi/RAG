# streamlit_app.py

import streamlit as st
import requests

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000/chat"

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="EV-Assist", layout="wide")
st.title("🚗 EV-Assist Chatbot")

# --------------------------------------------------
# Initialize chat memory
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# Render chat history
# --------------------------------------------------
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

        if turn["role"] == "assistant" and turn.get("citations"):
            with st.expander("Sources"):
                for i, citation in enumerate(turn["citations"], start=1):
                    st.markdown(f"[{i}] {citation}")

# --------------------------------------------------
# User input
# --------------------------------------------------
query = st.chat_input("Ask about Electric Vehicles")

if query:
    payload = {
        "query": query,
        "chat_history": st.session_state.chat_history,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Single source of truth
        st.session_state.chat_history = data["chat_history"]

        st.rerun()

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with FastAPI: {e}")
