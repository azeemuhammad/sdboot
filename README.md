# sdboot — Personal RAG Chatbot

**Live demo:** https://sdboot.streamlit.app/

**sdboot** is a Retrieval-Augmented Generation (RAG) chatbot with personal knowledge about:

1. **Muhammad Daniyal Azeem** — AI & Flutter developer, Lahore, BS Artificial Intelligence @ UMT  
2. **Muhammad Sher Khan** — Matta, Swat  

It answers only from the personal dataset (not generic Wikipedia data).

---

## 1. How it works (RAG pipeline)

Matches the standard RAG architecture:

```
1. Additional documents (JSONL knowledge base)
2. Encode → Embedding model (TF-IDF)
3. Index → Vector database (FAISS)
4. User Query → Encode
5. Similarity search → retrieve similar documents
6. Similar docs + Query → Prompt
7. LLM (Google Gemini) → Final response
```

| Component | Implementation |
|-----------|----------------|
| Document loading & preprocessing | JSONL load, clean whitespace, structured chunks |
| Embedding generation | TF-IDF (1–2 grams, L2-normalized) |
| Vector database | FAISS `IndexFlatIP` (cosine similarity) |
| Retrieval | Top-k + category/project score boosting |
| Prompt engineering | System prompt (identity + third-person rules) + context + history |
| LLM generation | Google Gemini API; offline fallback if key fails |
| History | Last 6 conversation turns |
| Interface | Streamlit web app (`app_streamlit.py`) + Gradio + CLI |
| Identity | Named chatbot **sdboot** |

The bot always speaks **as sdboot** (assistant), in **third person** about Daniyal or Sher Khan (never pretends to be them).

---

## 2. Dataset

| File | Content |
|------|---------|
| `sdboot_combined_knowledge.jsonl` | **Main dataset** — Daniyal (140) + Sher Khan (15) = 155 chunks |
| `daniyal_azeem_chatbot_knowledge.jsonl` | Original Daniyal-only knowledge |
| `Daniyal_Azeem_Chatbot_Knowledge_Base.pdf` | Human-readable Daniyal knowledge |

Data includes: profile, education, skills, projects (Clinic Portal, Wallpaper App, ML, ETL, LUMINA), contact, and Q&A pairs. All personal/custom — not generic dumps.

---

## 3. Project structure

```
sdboot/
├── app_streamlit.py              # Streamlit UI (deployed app)
├── app.py                        # Gradio UI
├── cli_chat.py                   # Terminal chat
├── rag_engine.py                 # Full RAG + Gemini
├── sdboot_combined_knowledge.jsonl
├── daniyal_azeem_chatbot_knowledge.jsonl
├── Daniyal_Azeem_Chatbot_Knowledge_Base.pdf
├── requirements.txt
├── README.md                     # This file
└── PROJECT_REPORT.md
```

---

## 4. Requirements

```
faiss-cpu>=1.7.0
numpy>=1.24.0
scikit-learn>=1.3.0
httpx>=0.27.0
streamlit>=1.28.0
gradio>=4.0.0
```

Install:

```bash
pip install -r requirements.txt
```

---

## 5. Local run (testing)

```bash
cd sdboot
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-api-key"
streamlit run app_streamlit.py
```

Open http://localhost:8501  

CLI:

```bash
python cli_chat.py
```

---

## 6. Deployment instructions (GitHub + Streamlit.app)

**Live link:** https://sdboot.streamlit.app/

### Steps that were followed

1. **Prepare project**  
   Complete source code, `requirements.txt`, README, dataset JSONL, and report.

2. **Push to GitHub**  
   - Create a **public** repository named `sdboot`.  
   - Upload all files from this folder (especially `app_streamlit.py`, `rag_engine.py`, `sdboot_combined_knowledge.jsonl`, `requirements.txt`).

3. **Deploy on Streamlit Community Cloud**  
   - Go to https://share.streamlit.io and sign in with GitHub.  
   - Click **New app**.  
   - Select repository: `sdboot`, branch: `main`.  
   - **Main file path:** `app_streamlit.py`  
   - **Secrets** (Advanced settings):
     ```toml
     GEMINI_API_KEY = "your-gemini-api-key"
     ```
   - Click **Deploy**.

4. **Public URL**  
   After build succeeds, the app is available at:  
   **https://sdboot.streamlit.app/**

Local-only run is not sufficient for submission; the public Streamlit.app link is required.

---

## 7. Example queries

- Who are you?  
- Who is Daniyal? / What is Clinic Portal? / What skills does Daniyal have?  
- Who is Sher Khan? / Where is Sher Khan from?  
- How can I contact Daniyal?  

---

## 8. Submission checklist

- [x] Complete source code  
- [x] `requirements.txt`  
- [x] README (this file) with deployment instructions  
- [x] Dataset files (`sdboot_combined_knowledge.jsonl`, etc.)  
- [x] Deployment instructions (GitHub + Streamlit)  
- [x] Working public link: https://sdboot.streamlit.app/  
