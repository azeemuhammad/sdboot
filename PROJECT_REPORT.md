# sdboot — RAG Chatbot Project Report

**Chatbot name:** sdboot  
**Live deployment:** https://sdboot.streamlit.app/

## Objective

Design and implement a RAG-based personal chatbot using custom personal data (not generic Wikipedia). The system retrieves relevant chunks and generates answers with an LLM, has a named identity, maintains history, and is publicly deployed.

## Dataset

Personal/custom data only:
- Muhammad Daniyal Azeem: profile, education (UMT), skills, projects, contact, 120+ Q&A
- Muhammad Sher Khan: profile, education, hometown (Matta, Swat)
- Combined file: `sdboot_combined_knowledge.jsonl` (155 documents)

## RAG implementation

1. Document loading & preprocessing (JSONL)
2. Embedding generation (TF-IDF)
3. Vector database (FAISS)
4. Retrieval (similarity search + boosting)
5. Prompt engineering (system prompt + context + history)
6. LLM generation (Google Gemini) with offline fallback
7. History maintenance
8. Interface: Streamlit (`app_streamlit.py`), Gradio, CLI
9. Identity: **sdboot** (answers in third person about the people)

## Deployment

- Platform: **Streamlit Community Cloud**
- Source: public GitHub repository
- Main file: `app_streamlit.py`
- Secret: `GEMINI_API_KEY`
- Public URL: **https://sdboot.streamlit.app/**

## Conclusion

sdboot meets all requirements: personal dataset, full RAG pipeline, LLM generation, named identity, history, working interface, and a functional public deployment link.
