---
title: NeuroACT Solitude Companion
emoji: 🌲
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# NeuroACT-Solitude: Controlled-Distance RAG Companion

Always-on, low-demand conversational companion for chronic schizoid withdrawal, autistic
burnout, and hikikomori-adjacent isolation. Runs on free ZeroGPU hardware via `transformers`
(falls back to CPU automatically off-Spaces), with retrieval over `knowledge_base.json` using
`BAAI/bge-small-en-v1.5`.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The model and embedder download automatically on first run.

## Deploy

From the repo root: `python scripts/deploy.py --hf-username YOU`. Free accounts get 2 ZeroGPU
Spaces; arch2 and arch3 use both.

## ZeroGPU rules this app follows

1. `import spaces` before `torch`.
2. Model loads at module scope and moves to CUDA eagerly.
3. Inference is wrapped in `@spaces.GPU(duration=30)`.

## Safety behavior

- `safety.py` runs before any model call. Text matching explicit suicide/self-harm patterns gets a
  fixed support-line message and no generation. Ambiguous distress phrases get a one-time support
  note appended to the reply.
- The engulfment/shutdown "circuit breaker" only shapes the prompt. It is not a crisis screen.
- Both are keyword heuristics, not clinical assessments. They need clinician review before
  public use. `safety.py` is mirrored in `arch1_liminal_space/crisis.js`.

## Your own model and data

Set the `NEURO_MODEL_ID` Space variable to your own merged model (deploy.py does this when you
pass `--merged-repo`). Extend `knowledge_base.json` to grow the retrieval corpus; new entries
need the same clinical review.

## Limitations

- Chat history lives in the browser session; there is no server-side persistence or export.
- The knowledge base is six short entries and is a placeholder for a reviewed corpus.
