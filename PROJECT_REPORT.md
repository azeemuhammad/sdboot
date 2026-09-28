# sdboot — RAG Chatbot Project Report

**Student:** Muhammad Daniyal Azeem  
**Chatbot name:** sdboot  
**Task:** Design & implement a RAG-based personal chatbot

---

## 1. Objective

Build a working Retrieval-Augmented Generation (RAG) system that answers questions about the developer using only his own personal dataset. The bot must have a clear identity (sdboot), maintain conversation history, use an LLM for generation, and be publicly deployed.

## 2. Dataset

- Source: `daniyal_azeem_chatbot_knowledge.jsonl` (140 documents)
- Content: profile, personal details, education, skills, six main projects, contact info, and 120+ curated Q&A pairs
- Format: one JSON object per line (`id`, `text`, `category`, optional `project_name`)
- Preprocessing: whitespace normalization, empty-line filtering
- No external generic corpora were used.

## 3. RAG Pipeline Implementation

| Step | Implementation |
|------|----------------|
| Document loading | JSONL reader → list of dicts |
| Embedding | TF-IDF (1–2 grams, L2-normalized) |
| Vector database | FAISS `IndexFlatIP` (cosine) |
| Retrieval | Top-k + category/project score boosting |
| LLM generation | Google Gemini (prompt + retrieved context + history) |
| Prompt engineering | System prompt defining identity + strict grounding rules |
| History | Sliding window of last 6 turns |
| Interface | Gradio web UI + CLI |
| Fallback | High-quality retrieval-only synthesizer if LLM unavailable |

## 4. Chatbot Identity

- Name: **sdboot**
- Role: Personal knowledge assistant for Muhammad Daniyal Azeem
- Tone: Helpful, direct, grounded in the knowledge base

## 5. Deployment

The system is ready for **Hugging Face Spaces** (Gradio SDK).

1. Create a Space → SDK = Gradio  
2. Upload all project files  
3. Add secret `GEMINI_API_KEY` = your Google Gemini key  
4. Live URL: `https://huggingface.co/spaces/<username>/sdboot`

Local runnable mode is also provided (`python app.py`).

## 6. How to run locally

```bash
cd sdboot
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
python app.py
```

## 7. Conclusion

sdboot fully satisfies the assignment requirements: personal dataset, complete RAG pipeline (load → embed → FAISS → retrieve → prompt → LLM), named identity, conversation history, working interface, and deployment instructions for a public platform.
