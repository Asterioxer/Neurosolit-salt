---
title: NeuroACT Solitude Companion
emoji: 🌲
colorFrom: green
colorTo: slate
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# NeuroACT-Solitude: Controlled-Distance CPU GGUF & BGE RAG Companion

Stateful, always-on, low-demand conversational companion for chronic
schizoid withdrawal, autistic burnout, and hikikomori-adjacent isolation.
Runs on CPU-Basic (2 vCPU / 16GB RAM, no GPU queue) via `llama-cpp-python`
and retrieves from a small fixed clinical knowledge base using
`BAAI/bge-small-en-v1.5` embeddings.

## Deploy

```bash
hf repos create <your-username>/neuroact-solitude \
  --type space --space-sdk gradio --flavor cpu-basic --public --exist-ok

git clone https://huggingface.co/spaces/<your-username>/neuroact-solitude
cd neuroact-solitude
cp /path/to/this/directory/* .
git add . && git commit -m "Deploy NeuroACT-Solitude" && git push
```

`requirements.txt` points at the prebuilt CPU wheel index for
`llama-cpp-python` so the Space builds in seconds instead of timing out on
a C++ compile.

## Before training a custom model

`app.py` defaults to `bartowski/Qwen2.5-1.5B-Instruct-GGUF`, a stock
quantized model. Swap `GGUF_REPO`/`GGUF_FILE` for your own export (see
`training/export_gguf.sh`) once you have one.

## The circuit breaker

`check_circuit_breaker()` is a regex/keyword pre-check that flags likely
engulfment-panic or verbal-shutdown language in the user's message and
overrides the system prompt accordingly before generation.

**This is a heuristic, not a safety system.** It will miss crisis signals
that don't match its patterns, and it may false-positive on ordinary text.
See repo root `docs/ROADMAP.md` Phase 1.5 — this is the component that
most needs clinical review before any public deployment.

## Hard grammar lock (optional, recommended before public deploy)

`llama-cpp-python` supports GBNF grammar constraints, which can block `?`
and `!` tokens at the sampling level rather than post-hoc:

```python
from llama_cpp import LlamaGrammar

DECLARATIVE_GBNF = r"""
root ::= sentence+
sentence ::= [A-Za-z0-9,;'":\-\(\)\s]+ "." [\s\n]*
"""
declarative_grammar = LlamaGrammar.from_string(DECLARATIVE_GBNF)

# pass grammar=declarative_grammar to llm.create_chat_completion(...)
```

This is not wired into `app.py` by default because it's a strict content
constraint that can make the model produce stilted or truncated text if the
grammar is too restrictive — test it against your own prompts before
enabling it unconditionally.

## Limitations

- Conversation history lives in server-side session memory for the
  duration of a session. No persistence, no export. See root
  `docs/ROADMAP.md` hardening backlog for the "let users export/delete
  their state" item.
- The RAG knowledge base (`CLINICAL_KNOWLEDGE_BASE` in `app.py`) is six
  hardcoded strings. It's a placeholder for a real retrieval corpus, not a
  finished knowledge base.
