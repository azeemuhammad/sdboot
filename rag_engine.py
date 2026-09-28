"""
sdboot RAG Engine
-----------------
Full Retrieval-Augmented Generation for the personal chatbot "sdboot".
Uses Muhammad Daniyal Azeem's personal knowledge base.

Pipeline (matches the provided architecture diagram):
  1. Load & preprocess personal documents (JSONL)
  2. Embedding generation (TF-IDF, offline)
  3. Vector database (FAISS)
  4. Similarity search / retrieval
  5. Prompt engineering + LLM generation (Google Gemini)
  6. Conversation history

Falls back to retrieval-only synthesis if no API key is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Gemini LLM helper
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def call_gemini(system_prompt: str, user_prompt: str, api_key: str = None) -> Optional[str]:
    """Call Google Gemini via REST. Returns generated text or None on failure."""
    key = api_key or GEMINI_API_KEY
    if not key:
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n---\n\n{user_prompt}"}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.9,
            "maxOutputTokens": 1024,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None
            return parts[0].get("text", "").strip()
    except Exception as e:
        print(f"[sdboot] Gemini call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main RAG class
# ---------------------------------------------------------------------------
class SDBootRAG:
    def __init__(
        self,
        data_path: str | Path,
        top_k: int = 5,
        min_score: float = 0.02,
        api_key: str = None,
    ):
        self.data_path = Path(data_path)
        self.top_k = top_k
        self.min_score = min_score
        self.api_key = api_key or GEMINI_API_KEY

        self.documents: List[Dict] = []
        self.texts: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self._ready = False

        self.history: List[Dict[str, str]] = []
        self.max_history = 6

    def load_documents(self) -> None:
        docs = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    if not text:
                        continue
                    text = " ".join(text.split())
                    docs.append(
                        {
                            "id": obj.get("id", f"doc_{len(docs)}"),
                            "text": text,
                            "category": obj.get("category", "general"),
                            "project_name": obj.get("project_name"),
                        }
                    )
                except json.JSONDecodeError:
                    continue

        self.documents = docs
        self.texts = [d["text"] for d in docs]
        print(f"[sdboot] Loaded {len(self.documents)} knowledge chunks.")

    def build_embeddings(self) -> None:
        if not self.texts:
            self.load_documents()

        custom_stops = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall", "can",
            "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
            "our", "you", "your", "he", "she", "they", "them", "his", "her",
        }
        self.vectorizer = TfidfVectorizer(
            max_features=8192,
            ngram_range=(1, 2),
            stop_words=list(custom_stops),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        matrix = self.vectorizer.fit_transform(self.texts)
        self.embeddings = normalize(matrix.toarray().astype(np.float32))
        print(f"[sdboot] TF-IDF embeddings ready: {self.embeddings.shape}")

    def build_index(self) -> None:
        if self.embeddings is None:
            self.build_embeddings()
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        self._ready = True
        print(f"[sdboot] FAISS index ready ({self.index.ntotal} vectors).")

    def ensure_ready(self) -> None:
        if not self._ready:
            self.load_documents()
            self.build_embeddings()
            self.build_index()

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        self.ensure_ready()
        k = k or self.top_k
        fetch_k = max(k * 4, 12)

        q_vec = self.vectorizer.transform([query])
        q_emb = normalize(q_vec.toarray().astype(np.float32))
        scores, indices = self.index.search(q_emb, fetch_k)

        results = []
        q_lower = query.lower()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < self.min_score:
                continue
            doc = self.documents[idx].copy()
            base = float(score)

            text_l = doc["text"].lower()
            if text_l.startswith("question:") and "answer:" in text_l:
                q_part = text_l.split("answer:")[0]
                if any(tok in q_part for tok in q_lower.split() if len(tok) > 3):
                    base += 0.25
                ans_part = text_l.split("answer:")[-1].strip()
                if len(ans_part) > 100:
                    base += 0.25
                elif len(ans_part) > 60:
                    base += 0.1
                if ans_part.startswith(("yes", "no")) and len(ans_part) < 90:
                    base -= 0.35

            proj = (doc.get("project_name") or "").lower()
            if proj and proj in q_lower:
                base += 0.35
            if doc.get("category") == "project" and any(
                w in q_lower for w in ("project", "built", "made", "clinic", "wallpaper", "lumina", "etl")
            ):
                base += 0.2

            cat = doc.get("category", "")
            if cat == "contact" and any(w in q_lower for w in ("contact", "email", "whatsapp", "phone", "reach")):
                base += 0.3
            if cat == "education" and any(w in q_lower for w in ("education", "university", "degree", "study", "school")):
                base += 0.25
            if cat == "skills" and any(w in q_lower for w in ("skill", "know", "tech", "stack")):
                base += 0.2

            doc["score"] = base
            results.append(doc)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    SYSTEM_PROMPT = """You are sdboot, a personal knowledge assistant.

You have information about TWO people in your knowledge base:
1) Muhammad Daniyal Azeem — AI & Flutter developer from Lahore, studying BS Artificial Intelligence at UMT.
2) Muhammad Sher Khan — from Matta, Swat (details in the context when relevant).

Rules:
- Answer ONLY using the provided CONTEXT. Never invent facts.
- If the user asks about Daniyal / Azeem / Flutter / Clinic Portal / UMT → use Daniyal's context.
- If the user asks about Sher Khan / Swat / Matta / Nawab Ali → use Sher Khan's context.
- If unclear who they mean, briefly answer for both or ask which person.
- Speak naturally and clearly. Keep answers concise (2–6 sentences).
- You are named sdboot. If asked who you are, say you are sdboot, the RAG assistant with knowledge about Daniyal Azeem and Sher Khan.
"""

    def _build_context(self, retrieved: List[Dict]) -> str:
        if not retrieved:
            return "No relevant documents found."
        parts = []
        for i, doc in enumerate(retrieved, 1):
            cat = doc.get("category", "")
            proj = doc.get("project_name")
            header = f"[{i}] ({cat}" + (f" | {proj}" if proj else "") + ")"
            parts.append(f"{header}\n{doc['text']}")
        return "\n\n".join(parts)

    def _history_text(self) -> str:
        if not self.history:
            return ""
        lines = []
        for turn in self.history[-6:]:
            role = "User" if turn["role"] == "user" else "sdboot"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def synthesize_fallback(self, query: str, retrieved: List[Dict]) -> str:
        """Offline fallback when Gemini is unavailable."""
        q_lower = query.lower().strip()

        if any(p in q_lower for p in ("who are you", "what are you", "your name", "who is sdboot")):
            return (
                "I'm **sdboot**, a personal RAG assistant with knowledge about Muhammad Daniyal Azeem and Muhammad Sher Khan. "
                "I know everything about Muhammad Daniyal Azeem — "
                "his background, skills, projects, education, and how to reach him. "
                "Ask me anything!"
            )
        if any(q_lower.startswith(g) for g in ("hi", "hello", "hey", "salam")):
            return (
                "Hey! I'm sdboot, here to tell you about Muhammad Daniyal Azeem. "
                "He's an AI & Flutter developer based in Lahore. "
                "What would you like to know?"
            )
        if not retrieved:
            return (
                "I don't have enough information in Daniyal's knowledge base for that. "
                "Try asking about his projects, skills, education or contact details."
            )

        if any(w in q_lower for w in ("project", "clinic", "wallpaper", "lumina", "etl", "prediction", "what is")):
            project_docs = [d for d in retrieved if d.get("category") == "project" or d.get("project_name")]
            if project_docs:
                project_docs.sort(key=lambda d: (d["score"], len(d["text"])), reverse=True)
                return project_docs[0]["text"]

        for doc in retrieved:
            text = doc["text"]
            if text.startswith("Question:") and "Answer:" in text:
                ans = text.split("Answer:", 1)[-1].strip()
                if len(ans) > 40:
                    return ans

        best = retrieved[0]["text"]
        if best.startswith("Question:") and "Answer:" in best:
            return best.split("Answer:", 1)[-1].strip()
        return best

    def generate(self, query: str, retrieved: List[Dict]) -> str:
        """LLM generation with proper prompt; falls back if needed."""
        context = self._build_context(retrieved)
        history = self._history_text()

        user_prompt = f"""CONTEXT (retrieved from Daniyal's personal knowledge base):
{context}

{f"RECENT CONVERSATION:{chr(10)}{history}{chr(10)}" if history else ""}
USER QUESTION: {query}

Answer as sdboot, using only the context above:"""

        if self.api_key:
            answer = call_gemini(self.SYSTEM_PROMPT, user_prompt, self.api_key)
            if answer:
                return answer

        return self.synthesize_fallback(query, retrieved)

    def chat(self, user_message: str) -> Tuple[str, List[Dict]]:
        user_message = user_message.strip()
        if not user_message:
            return "Please type a question.", []

        retrieved = self.retrieve(user_message)
        answer = self.generate(user_message, retrieved)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

        return answer, retrieved

    def clear_history(self) -> None:
        self.history = []


def create_rag(data_path: str | Path = None, api_key: str = None) -> SDBootRAG:
    if data_path is None:
        data_path = Path(__file__).parent / "sdboot_combined_knowledge.jsonl"
    rag = SDBootRAG(data_path, api_key=api_key)
    rag.ensure_ready()
    return rag
