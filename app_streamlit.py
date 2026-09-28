"""
sdboot — One RAG chatbot for both knowledge bases
(Muhammad Daniyal Azeem + Muhammad Sher Khan)
Ready for Streamlit Community Cloud.
"""

import os
from pathlib import Path
import streamlit as st

from rag_engine import create_rag

st.set_page_config(
    page_title="sdboot | Personal RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

@st.cache_resource
def load_rag():
    # Combined knowledge (Daniyal + Sher Khan)
    data_path = Path(__file__).parent / "sdboot_combined_knowledge.jsonl"
    if not data_path.exists():
        data_path = Path(__file__).parent / "daniyal_azeem_chatbot_knowledge.jsonl"
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return create_rag(data_path, api_key=api_key)

rag = load_rag()

with st.sidebar:
    st.title("sdboot")
    st.markdown("**One chatbot · Two people**")
    st.markdown(
        """
        **Muhammad Daniyal Azeem**  
        AI & Flutter · Lahore · UMT

        **Muhammad Sher Khan**  
        Matta, Swat
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **Try asking:**
        - Who is Daniyal?
        - What is Clinic Portal?
        - Who is Sher Khan?
        - Where is Sher Khan from?
        - Contact details for Daniyal
        """
    )
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        rag.clear_history()
        st.rerun()

st.title("sdboot")
st.caption("Personal RAG chatbot · Daniyal Azeem + Sher Khan")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Daniyal or Sher Khan…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, _ = rag.chat(prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
