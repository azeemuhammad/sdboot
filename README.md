# sdboot

**Personal RAG Chatbot for Muhammad Daniyal Azeem**

sdboot answers questions about Daniyal using his own curated knowledge base  
(profile, education, skills, projects, 120+ Q&A pairs).

It implements a complete **Retrieval-Augmented Generation** pipeline:

```
Documents → Encode (TF-IDF) → FAISS Vector DB
                                   ↑
User Query → Encode → Similarity Search → top-k docs
                                   ↓
              Prompt + Context → Google Gemini LLM → Final response
```

Conversation history is maintained. The bot has a clear personal identity: **sdboot**.

---

## Features (meets all assignment requirements)

| Requirement | Status |
|-------------|--------|
| Personal / custom dataset (not Wikipedia) | ✅ 140 chunks from Daniyal’s own data |
| Document loading & preprocessing | ✅ |
| Embedding generation | ✅ TF-IDF |
| Vector database | ✅ FAISS |
| Retrieval mechanism | ✅ Cosine + score boosting |
| LLM-based response generation | ✅ Google Gemini |
| Proper prompt engineering | ✅ System + context + history |
| History maintenance | ✅ Last 6 turns |
| Working chatbot interface | ✅ Gradio web UI + CLI |
| Personal identity / name | ✅ **sdboot** |
| Deployment | ✅ Hugging Face Spaces ready |

---

## Project structure

```
sdboot/
├── daniyal_azeem_chatbot_knowledge.jsonl   # personal knowledge base
├── Daniyal_Azeem_Chatbot_Knowledge_Base.pdf
├── rag_engine.py                           # full RAG + Gemini
├── app.py                                  # Gradio web UI
├── cli_chat.py                             # terminal chat
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## Quick start (local)

```bash
cd sdboot
pip install -r requirements.txt

# Set your Gemini API key (required for LLM generation)
export GEMINI_API_KEY="AQ.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Web interface
python app.py
# → open http://localhost:7860

# Or terminal
python cli_chat.py
```

Without the key the bot still works in retrieval-only mode.

---

## Deployment (Mandatory) — Hugging Face Spaces

This is the recommended free deployment platform for Gradio apps.

### Steps

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click **New Space**
   - Space name: `sdboot` (or any name)
   - SDK: **Gradio**
   - Visibility: Public
3. Upload **all files** from this folder (or push via git)
4. Go to **Settings → Variables and secrets**
   - Add a **Secret**:
     - Name: `GEMINI_API_KEY`
     - Value: your Gemini API key (`AQ.…`)
5. The Space will build automatically.
6. Your live link will be:
   ```
   https://huggingface.co/spaces/<your-username>/sdboot
   ```

Alternative platforms (also work):
- Streamlit Cloud (if you convert the UI)
- Render
- Railway

Local-only is **not** sufficient for the assignment — you must provide a public working link.

---

## Dataset

- 140 personal documents covering:
  - Profile & identity
  - Education (UMT, Matric, FSc)
  - Skills (Flutter, Dart, Firebase, Python, ML, SQL, C++)
  - Projects: Clinic Portal, Wallpaper App, Student Performance Prediction, ETL, Database Design, LUMINA
  - Contact & internship availability
  - 120+ ready Q&A pairs

All data is original / personal. No generic Wikipedia dumps.

---

## How the RAG works

1. **Load** – JSONL → list of documents  
2. **Embed** – TF-IDF (1–2 grams, L2-normalized)  
3. **Index** – FAISS IndexFlatIP (cosine similarity)  
4. **Retrieve** – top-k + category / project score boosting  
5. **Generate** – System prompt + retrieved context + history → Gemini  
6. **History** – sliding window of recent turns  

If the Gemini call fails, a high-quality retrieval-only synthesizer is used as fallback.

---

## Identity

The bot introduces itself as **sdboot**, Daniyal’s personal RAG assistant.  
It never invents facts outside the knowledge base.

---

## License / usage

Personal academic project. Free to use for learning and demonstration.
