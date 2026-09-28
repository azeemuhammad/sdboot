# SDBot – Personal RAG Chatbot

**SDBot** is a professional Retrieval-Augmented Generation (RAG) chatbot built for two people:

- **Muhammad Sher Khan**
- **Muhammad Daniyal Azeem**

It answers questions strictly from the combined personal knowledge base.  
If the information is not available, it replies:

> I don't have that information in my knowledge base.

---

## Features

- Personalized identity: **SDBot**
- Single combined knowledge file (JSONL)
- Document loading & preprocessing
- Embedding generation (`all-MiniLM-L6-v2`)
- Vector Database: **ChromaDB**
- Similarity search / Retrieval
- LLM response generation using **Google Gemini**
- Proper prompt engineering
- Chat history maintenance
- Clean Streamlit interface
- Works on Streamlit Cloud without path errors

---

## Project Structure

```
sdbot/
├── app_streamlit.py
├── requirements.txt
├── README.md
└── data/
    └── sdboot_combined_knowledge.jsonl
```

---

## How to Run Locally

```bash
cd sdbot
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_streamlit.py
```

---

## Deployment on Streamlit Cloud

1. Push the entire `sdbot` folder to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repository
4. Main file path: `app_streamlit.py`
5. Deploy

Make sure the `data/sdboot_combined_knowledge.jsonl` file is present in the repository.

---

## RAG Pipeline

1. Knowledge documents → `sdboot_combined_knowledge.jsonl`
2. Embedding model → `sentence-transformers/all-MiniLM-L6-v2`
3. Vector store → ChromaDB
4. Query encoding → Same embedding model
5. Similarity search → Top relevant documents
6. Prompt + Context → Google Gemini
7. Final response → SDBot
