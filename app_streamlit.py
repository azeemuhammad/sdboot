"""
sdboot — Personal RAG Chatbot (Streamlit version)
Ready for Streamlit Community Cloud deployment.
"""

import os
from pathlib import Path
import streamlit as st

from rag_engine import create_rag

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="sdboot | Daniyal's and Sher khan's RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ------------------------------------------------------------------
# Load RAG (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_rag():
    data_path = Path(__file__).parent / "daniyal_azeem_chatbot_knowledge.jsonl"
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return create_rag(data_path, api_key=api_key)

rag = load_rag()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("sdboot")
    st.markdown("**Personal RAG Chatbot**")
    st.markdown("Muhammad Daniyal Azeem")
    st.markdown("---")
    st.markdown(
        """
        **Ask about:**
        - Profile & Education
        - Skills (Flutter, Python, ML…)
        - Projects (Clinic Portal, Wallpaper App…)
        - Contact & Internship
        """
    )
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        rag.clear_history()
        st.rerun()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("sdboot")
st.caption("AI & Flutter developer · Lahore · BS Artificial Intelligence @ UMT")

# ------------------------------------------------------------------
# Chat history
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------
if prompt := st.chat_input("Ask about Daniyal’s projects, skills, education, contact…"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, _ = rag.chat(prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
