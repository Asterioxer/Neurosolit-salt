import re

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Stock quantized model by default -- swap for your own GGUF export
# (training/export_gguf.sh output) once you have one.
GGUF_REPO = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILE = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=2,
    verbose=False,
)

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Placeholder knowledge base -- six hardcoded strings. Replace with a real
# retrieval corpus before treating this as a finished RAG system.
CLINICAL_KNOWLEDGE_BASE = [
    "Schizoid Compromise & Controlled Distance (Erskine / Fairbairn): Never push for rapid emotional intimacy. Relate around objective, intellectual, or creative solitary projects so the permeable ego boundary feels safe from engulfment.",
    "Post-Disclosure Dissonance: When a solitary person shares personal information, they often experience immediate panic that the listener has gained power over them. Immediately validate stepping back and confirm zero expectations exist.",
    "Autistic Shutdown & Inertia (Lammers / Raymaker): During shutdown, spoken words and motor initiation go offline. Never force movement or ask test questions. Allow stationary recovery in a low-sensory room.",
    "BERMED Declarative Protocol (Murphy): Replace commands and questions with calm observations ('I notice...', 'I wonder...'). Always make room for the person's 'no' and extend wait time.",
    "NeuroACT & Behavioural Activation (Pahnke / Attwood): Dismantle the shame of unmasking, use cognitive defusion to untangle harsh internal criticism, and anchor wellbeing in deep, focused solitary interests.",
    "ACT-i Sleep & Sensory Pacing: Protect circadian recovery and sensory safety before cumulative demands stretch the internal stress band to its breaking point.",
]

doc_embeddings = embedder.encode(CLINICAL_KNOWLEDGE_BASE, normalize_embeddings=True)


def retrieve_clinical_guidance(query: str, top_k: int = 2) -> str:
    q_emb = embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = np.dot(doc_embeddings, q_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n".join([f"- {CLINICAL_KNOWLEDGE_BASE[i]}" for i in top_indices])


# NOTE: this pattern list is a starting point, not a validated crisis
# screen. See README.md "The circuit breaker" and root docs/ROADMAP.md
# Phase 1.5.
ENGULFMENT_PATTERNS = [
    r"\bsaid too much\b", r"\bregret telling\b", r"\bfeel exposed\b",
    r"\bwant to disappear\b", r"\bghost\b", r"\bsuffocat\w*\b",
    r"\btoo close\b", r"\bwords gone\b", r"\bcant move\b",
]


def check_circuit_breaker(text: str) -> str:
    lowered = text.lower().strip()
    if len(lowered) <= 8:
        return "SHUTDOWN_MINIMAL_BANDWIDTH"
    for pat in ENGULFMENT_PATTERNS:
        if re.search(pat, lowered):
            return "ENGULFMENT_CIRCUIT_BREAKER"
    return "NOMINAL"


def respond_companion(message, history, relational_mode, solitary_interest):
    breaker_state = check_circuit_breaker(message)
    rag_context = retrieve_clinical_guidance(message)

    system_instructions = (
        "Act as a non-pathologizing NeuroACT and Schizoid-Process informed companion.\n"
        f"Retrieved Clinical Principles:\n{rag_context}\n"
        "Core Rules:\n"
        "- Treat solitude, silence, and sequential departure relief as healthy homeostatic regulation.\n"
        "- Use declarative observations ('I notice...', 'One perspective is...').\n"
        "- Ask ZERO questions (never output '?') and issue ZERO imperative commands.\n"
    )

    if breaker_state == "ENGULFMENT_CIRCUIT_BREAKER":
        system_instructions += (
            "URGENT OVERRIDE: Engulfment dissonance detected. Validate protective withdrawal immediately. "
            "State clearly that nothing shared creates any tie or obligation, and pivot to impersonal neutrality."
        )
    elif breaker_state == "SHUTDOWN_MINIMAL_BANDWIDTH":
        system_instructions += (
            "URGENT OVERRIDE: Verbal shutdown detected. Output at most two short, calm declarative sentences requiring zero reply."
        )
    elif relational_mode == "Intellectual / Project Container (Schizoid-Safe)":
        system_instructions += (
            f"Frame all reflections objectively around the user's solitary project/interest: '{solitary_interest}'."
        )
    else:
        system_instructions += (
            f"Use NeuroACT cognitive defusion and sensory boundary pacing anchored in '{solitary_interest}'."
        )

    messages = [{"role": "system", "content": system_instructions}]
    for user_turn, bot_turn in history[-4:]:
        messages.append({"role": "user", "content": user_turn})
        if bot_turn:
            messages.append({"role": "assistant", "content": bot_turn})
    messages.append({"role": "user", "content": message})

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=60 if breaker_state != "NOMINAL" else 145,
        temperature=0.35,
        top_p=0.9,
        stream=True,
    )

    accumulated = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        accumulated += delta
        yield accumulated.replace("?", ".")


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NeuroACT-Solitude: Stateful CPU GGUF & Controlled-Distance RAG Companion")
    gr.Markdown(
        "_This is an unvalidated prototype, not a therapist replacement. "
        "If you are in crisis, please reach out to a crisis line or someone you trust._"
    )
    with gr.Row():
        rel_mode = gr.Dropdown(
            choices=[
                "Intellectual / Project Container (Schizoid-Safe)",
                "NeuroACT Defusion & Sensory Pacing",
            ],
            value="Intellectual / Project Container (Schizoid-Safe)",
            label="Controlled-Distance Framing Mode",
        )
        interest_input = gr.Textbox(
            value="Systems Architecture & Solitary Research",
            label="Focused Solitary Interest (Behavioural Activation Anchor)",
        )
    gr.ChatInterface(
        fn=respond_companion,
        additional_inputs=[rel_mode, interest_input],
    )

if __name__ == "__main__":
    demo.queue().launch()
