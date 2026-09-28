import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SDBot | Personal RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================================================
# CONSTANTS
# =========================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = "sdboot_combined_knowledge.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"

BOT_NAME = "SDBot"

SYSTEM_PROMPT = """You are SDBot, a professional personal assistant.

You have knowledge about only two people:
1. Muhammad Sher Khan
2. Muhammad Daniyal Azeem

STRICT RULES:
- Answer ONLY using the information given in the Context.
- If the answer is not present in the Context, reply exactly with:
  "I don't have that information in my knowledge base."
- Never invent, assume, or add extra details.
- Keep answers clear, natural, and professional.
- When the question is about a specific person, answer accordingly.
"""

# =========================================================
# GEMINI SETUP
# =========================================================
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "AQ.Ab8RN6IpbVJPeMr6FNDf77PqHkRVQz94kn7O7N1n38XhyfyPKA"
)

def get_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


# =========================================================
# FIND DATA FILE (works locally + Streamlit Cloud)
# =========================================================
def find_data_file() -> Path:
    possible_locations = [
        BASE_DIR / "data" / DATA_FILE,
        BASE_DIR / DATA_FILE,
        Path.cwd() / "data" / DATA_FILE,
        Path.cwd() / DATA_FILE,
        Path("/mount/src/sdboot/data") / DATA_FILE,
        Path("/mount/src/sdbot/data") / DATA_FILE,
        Path("/mount/src/sdboot") / DATA_FILE,
        Path("/mount/src/sdbot") / DATA_FILE,
    ]

    for path in possible_locations:
        if path.exists():
            return path

    return None


# =========================================================
# LOAD DOCUMENTS FROM JSONL
# =========================================================
def load_documents() -> List[Document]:
    data_path = find_data_file()

    if data_path is None:
        st.error(
            "❌ Knowledge file not found.\n\n"
            "Please make sure this file exists in your repository:\n"
            "`data/sdboot_combined_knowledge.jsonl`"
        )
        st.stop()

    documents = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text", "")
                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "id": item.get("id", ""),
                            "person": item.get("person", ""),
                            "category": item.get("category", ""),
                            "question": item.get("question", "")
                        }
                    )
                )
            except json.JSONDecodeError:
                continue

    if not documents:
        st.error("No valid documents found in the knowledge file.")
        st.stop()

    return documents


# =========================================================
# VECTOR STORE
# =========================================================
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Always rebuild for reliability on Streamlit Cloud
    docs = load_documents()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    return vectorstore


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# =========================================================
# RAG + LLM RESPONSE
# =========================================================
def generate_response(query: str, history: List[Dict], vectorstore) -> str:
    # Retrieve top relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    # Build short conversation history
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "SDBot"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"""{SYSTEM_PROMPT}

Context from knowledge base:
{context}

Conversation history:
{history_text}

User Question: {query}

Answer:"""

    try:
        model = get_gemini_model()
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 600,
            }
        )
        answer = response.text.strip()
        return answer if answer else "I don't have that information in my knowledge base."
    except Exception:
        # Safe fallback
        if context:
            return context.split("Answer:")[-1].strip() if "Answer:" in context else context[:300]
        return "I don't have that information in my knowledge base."


# =========================================================
# UI
# =========================================================
def main():
    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("## 🤖 SDBot")
        st.caption("Personal RAG Assistant")
        st.divider()

        st.markdown("### Knowledge Base")
        st.success("Muhammad Sher Khan")
        st.success("Muhammad Daniyal Azeem")
        st.caption("Combined personal knowledge loaded")

        st.divider()
        st.markdown("### About")
        st.info(
            "SDBot answers questions **only** from the personal data of "
            "Muhammad Sher Khan and Muhammad Daniyal Azeem.\n\n"
            "If the information is not available, it will clearly say so."
        )

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("RAG Pipeline")
        st.caption("Embedding → ChromaDB → Gemini")

    # ---------- Main Area ----------
    st.title("SDBot")
    st.caption("Ask anything about Muhammad Sher Khan or Muhammad Daniyal Azeem")

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am **SDBot**.\n\n"
                    "I can answer questions about **Muhammad Sher Khan** and "
                    "**Muhammad Daniyal Azeem** using their personal knowledge base.\n\n"
                    "What would you like to know?"
                )
            }
        ]

    # Load vector store
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore()

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about name, education, projects, skills..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                response = generate_response(
                    prompt,
                    st.session_state.messages,
                    st.session_state.vectorstore
                )
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
