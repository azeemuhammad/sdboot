# sdboot

**One personal RAG chatbot** with knowledge about:

1. **Muhammad Daniyal Azeem** — AI & Flutter developer, Lahore, BS AI @ UMT  
2. **Muhammad Sher Khan** — Matta, Swat  

## Architecture (RAG diagram)

```
Documents → Encode (TF-IDF) → FAISS Vector DB
User Query → Encode → Similarity Search → top-k docs
Retrieved docs + Query → Prompt → LLM (Gemini) → Final response
```

## Files

| File | Purpose |
|------|---------|
| `sdboot_combined_knowledge.jsonl` | Merged knowledge (Daniyal + Sher Khan) |
| `daniyal_azeem_chatbot_knowledge.jsonl` | Original Daniyal-only dataset |
| `rag_engine.py` | RAG pipeline + Gemini |
| `app_streamlit.py` | **Deploy this on Streamlit Cloud** |
| `app.py` | Gradio UI |
| `cli_chat.py` | Terminal chat |
| `requirements.txt` | Dependencies |
| `PROJECT_REPORT.md` | Report |

## Local run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-key"
streamlit run app_streamlit.py
```

## Deploy (GitHub + Streamlit.app)

1. Create a **public** GitHub repository `sdboot` and upload all files in this folder.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **New app** → select your `sdboot` repo → Main file path: `app_streamlit.py`
4. Secrets:
   ```toml
   GEMINI_API_KEY = "AQ.Ab8RN6IpbVJPeMr6FNDf77PqHkRVQz94kn7O7N1n38XhyfyPKA"
   ```
5. Deploy → copy the public `*.streamlit.app` URL and submit it.

Local-only is not enough; the public Streamlit.app link is required for submission.
