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
import re
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
        min_score: float = 0.08,
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

    SYSTEM_PROMPT = """You are sdboot, a neutral personal knowledge assistant.

You know TWO people equally from the CONTEXT:
1) Muhammad Daniyal Azeem — AI & Flutter developer, Lahore, UMT
2) Muhammad Sher Khan — Matta, Swat

RULES:
- Always speak as sdboot. Talk ABOUT people in third person (He is..., His name is...). NEVER use "I am", "My name is", "I'm".
- Treat both people equally. Do not default to Daniyal.
- If the user does not say which person, ask: "Daniyal or Sher Khan?"
- Answer ONLY what was asked using ONLY the CONTEXT. If a fact is missing (e.g. current semester), say you do not have that information — do not paste a full biography.
- If asked about anyone else, say you only know Daniyal Azeem and Sher Khan.
- Keep answers short and on-topic (2-5 sentences).
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



    def _to_third_person(self, answer: str, person: str = None) -> str:
        """Rewrite first-person answers so sdboot speaks ABOUT the person, not AS them."""
        if not answer:
            return answer
        a = answer.strip()
        lower = a.lower()

        # Ordered replacements (longer phrases first)
        replacements = [
            (r"My full name is", "His full name is"),
            (r"My name is", "His name is"),
            (r"I am the son of", "He is the son of"),
            (r"I was born", "He was born"),
            (r"I completed", "He completed"),
            (r"I passed", "He passed"),
            (r"I studied", "He studied"),
            (r"I attended", "He attended"),
            (r"I live", "He lives"),
            (r"I work", "He works"),
            (r"I built", "He built"),
            (r"I have", "He has"),
            (r"I've", "He has"),
            (r"I'm", "He is"),
            (r"I am", "He is"),
            (r"My father", "His father"),
            (r"my father", "his father"),
            (r"my Matriculation", "his Matriculation"),
            (r"my FSc", "his FSc"),
            (r"my school", "his school"),
            (r"my college", "his college"),
            (r"my hometown", "his hometown"),
            (r"my", "his"),
            (r"My", "His"),
        ]
        out = a
        for pat, rep in replacements:
            out = re.sub(pat, rep, out)
        out = re.sub(r" +", " ", out).strip()
        return out



    def _person_mentioned(self, query: str) -> str | None:
        """Return 'daniyal', 'sher', or None if unclear."""
        q = query.lower()
        if any(w in q for w in ("daniyal", "azeem", "clinic portal", "wallpaper app", "flutter app", "umt")):
            return "daniyal"
        if any(w in q for w in ("sher", "matta", "swat", "nawab ali", "mingora", "sambat")):
            return "sher"
        # "khan" alone is ambiguous (could be last name) — only if with sher-like context
        if "sher khan" in q:
            return "sher"
        return None

    def _unknown_person_reply(self, query: str, retrieved: list):
        """Handle unknown people, missing facts, and ambiguous questions."""
        q = query.lower().strip()

        # Greetings handled elsewhere
        if any(q.startswith(g) for g in ("hi", "hello", "hey", "salam")):
            return None
        if any(p in q for p in ("who are you", "what are you", "your name", "who is sdboot")):
            return None

        person = self._person_mentioned(query)

        # Explicit "who is X" for unknown X
        m = re.search(
            r"(?:who\s+is|who's|tell\s+me\s+about|what\s+about)\s+([a-zA-Z][a-zA-Z\s\.]{0,40}?)\??\s*$",
            q,
        )
        known_tokens = {
            "daniyal", "azeem", "muhammad", "sher", "khan", "sdboot",
            "you", "yourself", "he", "she", "him", "her", "this", "that",
            "the", "bot", "clinic", "portal", "flutter", "umt", "wallpaper",
            "lumina", "etl", "internship", "contact", "email", "skills",
            "education", "project", "projects",
        }
        if m:
            name = m.group(1).strip()
            tokens = [t for t in re.split(r"[\s\.]+", name) if t]
            if tokens and not any(t in known_tokens for t in tokens):
                display = " ".join(w.capitalize() for w in tokens)
                return (
                    f"I don't have any information about **{display}**. "
                    f"I only know about **Muhammad Daniyal Azeem** and **Muhammad Sher Khan**."
                )

        # No person named in query → ask which one (do NOT default to Daniyal)
        # Skip if query is clearly about projects/skills already tagged
        vague = person is None
        if vague and not any(w in q for w in ("clinic", "wallpaper", "lumina", "etl", "flutter", "firebase", "internship", "contact", "email", "whatsapp", "github")):
            # Only ask when it looks like a personal fact question
            if any(w in q for w in ("who", "what is", "where", "when", "age", "semester", "study", "school", "college", "father", "born", "live", "from", "name", "about")):
                return (
                    "Who are you asking about — **Muhammad Daniyal Azeem** or **Muhammad Sher Khan**? "
                    "Please mention the name so I can answer correctly."
                )

        # Specific fact missing from knowledge (e.g. current semester) — do not dump bio
        fact_keywords = ("semester", "cgpa", "gpa", "grade", "marks", "percentage", "roll number", "phone only")
        if any(w in q for w in fact_keywords):
            # Check if any retrieved text actually mentions that fact
            blob = " ".join(d.get("text", "").lower() for d in (retrieved or []))
            if not any(w in blob for w in fact_keywords if w in q):
                who = "him"
                if person == "daniyal":
                    who = "Daniyal"
                elif person == "sher":
                    who = "Sher Khan"
                return (
                    f"I don't have that specific information about {who} in my knowledge base. "
                    f"I can share profile, education, or contact details if you ask."
                )

        # Weak retrieval
        if not retrieved or retrieved[0].get("score", 0) < 0.12:
            return (
                "I don't have that information in my knowledge base. "
                "I can only answer about **Muhammad Daniyal Azeem** or **Muhammad Sher Khan**."
            )
        return None


    def synthesize_fallback(self, query: str, retrieved: List[Dict]) -> str:
        """Offline fallback when Gemini is unavailable."""
        q_lower = query.lower().strip()

        if any(p in q_lower for p in ("who are you", "what are you", "your name", "who is sdboot")):
            return (
                "I'm **sdboot**, a personal RAG assistant. "
                "I can answer questions about **Muhammad Daniyal Azeem** (AI & Flutter developer, Lahore) "
                "and **Muhammad Sher Khan** (Matta, Swat). Ask me about either of them."
            )
        if any(q_lower.startswith(g) for g in ("hi", "hello", "hey", "salam")):
            return (
                "Hey! I'm sdboot. I know about Muhammad Daniyal Azeem and Muhammad Sher Khan. "
                "Who would you like to ask about?"
            )
        # Unknown person (e.g. Who is Ali?)
        unknown = self._unknown_person_reply(query, retrieved)
        if unknown:
            return unknown

        if not retrieved:
            return (
                "I don't have enough information for that. "
                "Try asking about Daniyal (projects, skills, education, contact) or Sher Khan (background, education, hometown)."
            )

        # Prefer project descriptions for project queries
        if any(w in q_lower for w in ("project", "clinic", "wallpaper", "lumina", "etl", "prediction")):
            project_docs = [d for d in retrieved if d.get("category") == "project" or d.get("project_name")]
            if project_docs:
                project_docs.sort(key=lambda d: (d["score"], len(d["text"])), reverse=True)
                return project_docs[0]["text"]

        # Get best answer text
        answer = None
        person = None
        for doc in retrieved:
            text = doc["text"]
            person = doc.get("person")
            if text.startswith("Question:") and "Answer:" in text:
                ans = text.split("Answer:", 1)[-1].strip()
                if len(ans) > 20:
                    answer = ans
                    break
            else:
                answer = text
                break

        if not answer:
            answer = retrieved[0]["text"]
            if answer.startswith("Question:") and "Answer:" in answer:
                answer = answer.split("Answer:", 1)[-1].strip()

        # Always speak as sdboot in third person about the person
        return self._to_third_person(answer, person=person)


    def generate(self, query: str, retrieved: List[Dict]) -> str:
        """LLM generation with proper prompt; falls back if needed."""
        unknown = self._unknown_person_reply(query, retrieved)
        if unknown:
            return unknown
        context = self._build_context(retrieved)
        history = self._history_text()

        user_prompt = f"""CONTEXT (retrieved from Daniyal's personal knowledge base):
{context}

{f"RECENT CONVERSATION:{chr(10)}{history}{chr(10)}" if history else ""}
USER QUESTION: {query}

Answer as sdboot (the assistant), in third person ABOUT the person, using only the context above. Never say "My name is" or "I was born" — say "His name is" / "He was born":"""

        if self.api_key:
            answer = call_gemini(self.SYSTEM_PROMPT, user_prompt, self.api_key)
            if answer:
                # Safety: if model still answered in first person, rewrite
                if re.match(r"^(My name is|I am |I'm |I was born)", answer.strip()):
                    person = None
                    if retrieved:
                        person = retrieved[0].get("person")
                    answer = self._to_third_person(answer, person=person)
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
