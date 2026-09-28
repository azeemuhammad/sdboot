"""
sdboot — Personal RAG Chatbot for Muhammad Daniyal Azeem
========================================================
Gradio web interface. Ready for Hugging Face Spaces / local.

Run locally:
    python app.py

Deploy on Hugging Face Spaces:
    - Create a new Space (Gradio SDK)
    - Upload all files in this folder
    - Add secret: GEMINI_API_KEY = your key
"""

import os
from pathlib import Path

import gradio as gr

from rag_engine import create_rag, SDBootRAG

# ------------------------------------------------------------------
# API key (from environment or Hugging Face Space secrets)
# ------------------------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

DATA_PATH = Path(__file__).parent / "sdboot_combined_knowledge.jsonl"
rag: SDBootRAG = create_rag(DATA_PATH, api_key=API_KEY)

SYSTEM_INTRO = """
**sdboot** — Daniyal’s personal knowledge assistant

I answer questions about **Muhammad Daniyal Azeem**:
• Profile, education & personal background  
• Skills (Flutter, Python, Firebase, ML, SQL…)  
• Projects (Clinic Portal, Wallpaper App, Student Performance Prediction, ETL, LUMINA…)  
• Contact & availability for internships / freelance  

Powered by a full RAG pipeline + Google Gemini LLM over his personal dataset.
"""


def respond(message: str, history: list) -> tuple[str, list]:
    if not message or not message.strip():
        return "", history

    answer, sources = rag.chat(message)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history


def clear_chat():
    rag.clear_history()
    return None, []


custom_css = """
.gradio-container { max-width: 860px !important; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="sdboot — Daniyal’s RAG Chatbot",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"),
    css=custom_css,
) as demo:

    gr.Markdown(
        """
        # sdboot
        **Personal RAG Chatbot for Muhammad Daniyal Azeem**  
        *AI & Flutter developer · Lahore, Pakistan · BS Artificial Intelligence @ UMT*
        """
    )

    with gr.Accordion("About this bot", open=False):
        gr.Markdown(SYSTEM_INTRO)
        gr.Markdown(
            """
            **Architecture**  
            Documents → TF-IDF Embeddings → FAISS Vector DB → Similarity Search  
            → Prompt + Context → **Google Gemini LLM** → Final response  
            (Matches the classic RAG pipeline)
            """
        )

    chatbot = gr.Chatbot(
        height=480,
        show_label=False,
        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=sdboot"),
        type="messages",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about Daniyal’s projects, skills, education, contact…",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

    gr.Examples(
        examples=[
            "Who are you?",
            "Tell me about Daniyal",
            "What is the Clinic Portal project?",
            "What skills does he have?",
            "Is he looking for an internship?",
            "How can I contact him?",
            "What is his education background?",
            "Does he know Flutter and Firebase?",
        ],
        inputs=msg,
        label="Try these",
    )

    gr.Markdown(
        "<center style='opacity:0.6;font-size:0.85em'>"
        "Built with personal knowledge base · Full RAG pipeline · Google Gemini"
        "</center>"
    )

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, None, [msg, chatbot])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
