try:
    import spaces  # ZeroGPU rule 1: import before torch.
except ImportError:
    # No-op mock for local development and non-ZeroGPU environments
    class _SpacesMock:
        @staticmethod
        def GPU(duration=60):
            def decorator(fn):
                return fn
            return decorator
    spaces = _SpacesMock()

import json
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety import CRISIS_MESSAGE, WATCH_NOTE, assess_risk

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Hardware-adaptive loading (same pattern as arch2):
#   ZeroGPU / CUDA -> fp16 on the GPU      CPU-only (laptop) -> fp32 on the CPU
# Set NEURO_MODEL_ID to your own merged model (deploy.py does this with --merged-repo).
# ---------------------------------------------------------------------------
USE_CUDA = bool(os.getenv("SPACES_ZERO_GPU")) or torch.cuda.is_available()
MODEL_ID = os.getenv("NEURO_MODEL_ID") or "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if USE_CUDA:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print(f"[neuroact] model={MODEL_ID} cuda={USE_CUDA}")

# ---------------------------------------------------------------------------
# RAG over a small fixed knowledge base (knowledge_base.json), embedded on CPU.
# Anything you add there needs the same clinician review as the rest.
# ---------------------------------------------------------------------------
CLINICAL_KNOWLEDGE_BASE = json.loads((HERE / "knowledge_base.json").read_text(encoding="utf-8"))
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

try:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    doc_embeddings = embedder.encode(CLINICAL_KNOWLEDGE_BASE, normalize_embeddings=True)
except Exception as exc:  # noqa: BLE001 - app must still start without retrieval
    print(f"[neuroact] embedder unavailable ({exc}); falling back to first entries.")
    embedder = None
    doc_embeddings = None


def retrieve_clinical_guidance(query: str, top_k: int = 2) -> str:
    if embedder is None:
        idx = list(range(min(top_k, len(CLINICAL_KNOWLEDGE_BASE))))
    else:
        q_emb = embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        idx = np.argsort(np.dot(doc_embeddings, q_emb))[::-1][:top_k]
    return "\n".join(f"- {CLINICAL_KNOWLEDGE_BASE[i]}" for i in idx)


# Engulfment / shutdown heuristics shape the system prompt. They are NOT the
# crisis screen -- that is safety.assess_risk(), which runs first.
ENGULFMENT_PATTERNS = [
    r"\bsaid too much\b", r"\bregret telling\b", r"\bfeel exposed\b",
    r"\bwant to disappear\b", r"\bghost\b", r"\bsuffocat\w*\b",
    r"\btoo close\b", r"\bwords gone\b", r"\bcan'?t move\b",
]

COMMON_GREETINGS = {
    "hi", "hello", "hey", "hiya", "good morning", "good evening", "good afternoon",
    "greetings", "yo", "sup", "howdy", "thanks", "thank you", "ok", "okay",
}


def check_circuit_breaker(text: str) -> str:
    lowered = text.lower().strip()
    if lowered in COMMON_GREETINGS:
        return "NOMINAL"
    if len(lowered) <= 8:
        return "SHUTDOWN_MINIMAL_BANDWIDTH"
    for pat in ENGULFMENT_PATTERNS:
        if re.search(pat, lowered):
            return "ENGULFMENT_CIRCUIT_BREAKER"
    return "NOMINAL"


def enforce_declarative_lock(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    safe = [s for s in sentences if not s.endswith("?")]
    cleaned = " ".join(safe) if safe else "This space remains quiet and unhurried. No reply is needed."
    return cleaned.replace("?", ".").replace("!", ".")


def _generate(messages, max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.35,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return enforce_declarative_lock(text)


# ZeroGPU rule 3: tight duration on the GPU function (no-op elsewhere).
@spaces.GPU(duration=30)
def _generate_gpu(messages, max_new_tokens: int) -> str:
    return _generate(messages, max_new_tokens)


def respond_companion(message, history, relational_mode, solitary_interest):
    # 1. Hard crisis gate: no model call, fixed text.
    risk = assess_risk(message)
    if risk == "crisis":
        return CRISIS_MESSAGE

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
    # ChatInterface(type="messages") passes history as role/content dicts.
    for turn in (history or [])[-6:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("content"):
            content = str(turn["content"])
            # Clinical safeguard: do not feed past crisis turns or crisis notices into LLM context
            if assess_risk(content) == "crisis" or ("14416" in content and "Tele-MANAS" in content):
                continue
            messages.append({"role": turn["role"], "content": content})
    messages.append({"role": "user", "content": message})

    max_new_tokens = 60 if breaker_state != "NOMINAL" else 145
    reply = _generate_gpu(messages, max_new_tokens) if USE_CUDA else _generate(messages, max_new_tokens)
    if risk == "watch":
        reply = reply + "\n\n" + WATCH_NOTE
    return reply


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NeuroACT-Solitude: Controlled-Distance RAG Companion")
    gr.Markdown(
        "_Unvalidated prototype, not a therapist replacement. Writing that sounds like thoughts of "
        "ending life or self-harm is not answered by the model; a fixed list of support lines is shown instead._"
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
        type="messages",
        additional_inputs=[rel_mode, interest_input],
        examples=[
            [
                "Been in my room for three days working on my code. Society feels loud and impossible.",
                "Intellectual / Project Container (Schizoid-Safe)",
                "Systems Architecture & Solitary Research",
            ],
            [
                "Someone invited me to an event and I don't know how to step back without feeling guilty.",
                "NeuroACT Defusion & Sensory Pacing",
                "Systems Architecture & Solitary Research",
            ],
            [
                "Words gone. Too loud earlier. Everyone finally left the house.",
                "Intellectual / Project Container (Schizoid-Safe)",
                "Systems Architecture & Solitary Research",
            ],
        ],
        api_name="chat",
    )
    with gr.Accordion("Clinical Transparency: Underlying Principles (Knowledge Base)", open=False):
        for item in CLINICAL_KNOWLEDGE_BASE:
            gr.Markdown(f"- {item}")

if __name__ == "__main__":
    demo.queue().launch()
