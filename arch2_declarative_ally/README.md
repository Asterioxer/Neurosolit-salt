---
title: Declarative Ally PDA Translator
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# Declarative-Ally: BERMED Demand De-Escalator & Shutdown Proxy

Translates incoming high-demand text (emails, instructions, self-critical checklists)
into BERMED-style declarative language, and provides zero-typing boundary cards for
acute shutdown. Built for PDA autism and autistic shutdown/inertia.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Models download automatically on first run. With a CUDA GPU it loads Qwen2.5-3B;
CPU-only it loads Qwen2.5-1.5B (slower, but works). Set `NEURO_MODEL_ID` to use your
own model.

## Deploy

From the repo root: `python scripts/deploy.py --hf-username YOU`. It tries ZeroGPU first and
falls back to free CPU-Basic if your account cannot host ZeroGPU.

## ZeroGPU rules this app follows

1. `import spaces` before `torch`.
2. Model loads at module scope and moves to CUDA eagerly.
3. Inference is wrapped in `@spaces.GPU(duration=20)`.

## Safety behavior

- `safety.py` runs before any model call. Text matching explicit suicide/self-harm patterns is
  not rewritten; a fixed support-line message is returned instead. Ambiguous distress phrases
  get a one-time support note appended.
- This is a keyword screen, not a clinical assessment. It needs clinician review before
  public use. `safety.py` is mirrored in `arch1_liminal_space/crisis.js`.

## Limitations

- `enforce_declarative_lock()` is a post-hoc regex filter, not a generation-time constraint.
- Not clinically validated.
