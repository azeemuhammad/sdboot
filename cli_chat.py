#!/usr/bin/env python3
"""
sdboot CLI – quick terminal interface
Usage:  python cli_chat.py
"""

from pathlib import Path
from rag_engine import create_rag

def main():
    data = Path(__file__).parent / "daniyal_azeem_chatbot_knowledge.jsonl"
    print("Loading sdboot RAG engine…")
    rag = create_rag(data)
    print("Ready. Type your question (or 'quit' / 'exit' to leave, 'clear' to reset history).\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user.lower() == "clear":
            rag.clear_history()
            print("(history cleared)\n")
            continue

        answer, sources = rag.chat(user)
        print(f"\nsdboot: {answer}\n")

        # Optional debug: show sources
        # for s in sources:
        #     print(f"  ↳ [{s.get('score',0):.2f}] {s.get('id')} ({s.get('category')})")

if __name__ == "__main__":
    main()
