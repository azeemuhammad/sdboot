import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict

# LangChain & RAG
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Gemini
import google.generativeai as genai

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="SDBot | Personal RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====================== CONSTANTS ======================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"

BOT_NAME = "SDBot"

SYSTEM_INSTRUCTION = """You are SDBot, a helpful personal assistant that answers questions ONLY about two people:
1. Muhammad Sher Khan
2. Muhammad Daniyal Azeem

Rules you must follow strictly:
- Use ONLY the information provided in the Context below.
- If the answer is not present in the Context, reply exactly: "I don't have that information in my knowledge base."
- Never invent or assume any details.
- Speak naturally and professionally.
- When answering about a specific person, clearly mention the name if needed.
- Keep answers concise and accurate.
"""

# ====================== GEMINI SETUP ======================
# Put your Gemini API key here or use environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6IpbVJPeMr6FNDf77PqHkRVQz94kn7O7N1n38XhyfyPKA")

def init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

# ====================== DATA LOADING ======================
def load_documents() -> List[Document]:
    """Load both personal datasets and convert them into LangChain Documents."""
    documents = []

    # Try multiple possible locations (works both locally and on Streamlit Cloud)
    possible_dirs = [
        DATA_DIR,                              # normal: sdbot/data/
        BASE_DIR,                              # files in same folder as app
        Path.cwd() / "data",                   # current working directory
        Path.cwd(),                            # current working directory root
        Path("/mount/src/sdboot/data"),        # Streamlit Cloud common path
        Path("/mount/src/sdbot/data"),
    ]

    sher_path = None
    daniyal_path = None

    for d in possible_dirs:
        s = d / "sher_khan_qa_dataset.json"
        dan = d / "daniyal_azeem_qa_dataset.json"
        if s.exists() and dan.exists():
            sher_path = s
            daniyal_path = dan
            break

    if sher_path is None or daniyal_path is None:
        st.error("❌ Dataset files not found! Please make sure the 'data' folder with both JSON files is uploaded to your GitHub repository.")
        st.stop()

    # --- Sher Khan ---
    with open(sher_path, "r", encoding="utf-8") as f:
        sher_data = json.load(f)

    for i, item in enumerate(sher_data):
        text = f"Person: Muhammad Sher Khan\nQuestion: {item['question']}\nAnswer: {item['answer']}"
        documents.append(Document(
            page_content=text,
            metadata={
                "person": "Muhammad Sher Khan",
                "source": "sher_khan_dataset",
                "id": f"sher_{i}",
                "question": item["question"]
            }
        ))

    # --- Daniyal Azeem ---
    with open(daniyal_path, "r", encoding="utf-8") as f:
        daniyal_data = json.load(f)

    for i, item in enumerate(daniyal_data):
        text = f"Person: Muhammad Daniyal Azeem\nQuestion: {item['question']}\nAnswer: {item['answer']}"
        documents.append(Document(
            page_content=text,
            metadata={
                "person": "Muhammad Daniyal Azeem",
                "source": "daniyal_azeem_dataset",
                "id": f"daniyal_{i}",
                "question": item["question"]
            }
        ))

    return documents


# ====================== VECTOR STORE ======================
@st.cache_resource(show_spinner="Building knowledge base...")
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    if CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings
        )
    else:
        docs = load_documents()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(CHROMA_PATH)
        )
    return vectorstore


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


# ====================== RAG + LLM ======================
def generate_response(query: str, chat_history: List[Dict], vectorstore) -> str:
    """Full RAG pipeline: Retrieve → Prompt → Gemini → Answer"""

    # 1. Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    # 2. Build conversation history text
    history_text = ""
    for msg in chat_history[-6:]:  # last 3 turns
        role = "User" if msg["role"] == "user" else "SDBot"
        history_text += f"{role}: {msg['content']}\n"

    # 3. Create final prompt
    full_prompt = f"""{SYSTEM_INSTRUCTION}

Context from knowledge base:
{context}

Conversation so far:
{history_text}

User Question: {query}

Your Answer:"""

    # 4. Call Gemini
    try:
        model = init_gemini()
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 512,
            }
        )
        answer = response.text.strip()
        return answer
    except Exception as e:
        # Fallback if API fails
        if context and "Answer:" in context:
            # Simple extraction fallback
            for line in context.split("\n"):
                if line.startswith("Answer:"):
                    return line.replace("Answer:", "").strip()
        return "I don't have that information in my knowledge base."


# ====================== UI ======================
def main():
    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("## 🤖 SDBot")
        st.caption("Personal RAG Assistant")
        st.divider()

        st.markdown("### Knowledge Base")
        st.success("Muhammad Sher Khan")
        st.success("Muhammad Daniyal Azeem")
        st.caption("Two personal datasets loaded")

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
                "content": "Hello! I am **SDBot**. I can answer questions about **Muhammad Sher Khan** and **Muhammad Daniyal Azeem**. What would you like to know?"
            }
        ]

    # Load vector store once
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore()

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about name, education, projects, skills..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
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
