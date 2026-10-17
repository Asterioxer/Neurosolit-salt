---
title: Declarative Ally PDA Translator
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Declarative-Ally: ZeroGPU BERMED Demand De-Escalator & Shutdown Proxy

Translates incoming high-demand text (emails, instructions, self-critical
internal checklists) into Linda K. Murphy's BERMED declarative language, and
provides zero-typing boundary cards for use during acute shutdown when
spoken/written language generation is inaccessible.

Built for Pervasive Drive for Autonomy (PDA) autism and acute autistic
shutdown/inertia.

## Deploy

```bash
hf repos create <your-username>/declarative-ally \
  --type space --space-sdk gradio --flavor zero-a10g --public --exist-ok

git clone https://huggingface.co/spaces/<your-username>/declarative-ally
cd declarative-ally
cp /path/to/this/directory/* .
git add . && git commit -m "Deploy Declarative-Ally" && git push
```

## ZeroGPU rules this app follows

1. `import spaces` is the first import in `app.py`, before `torch`.
2. Model and tokenizer load at module scope and move to `.to("cuda")`
   eagerly, so ZeroGPU can intercept, pack, and stream weights on first
   `@spaces.GPU` call.
3. The inference function is decorated `@spaces.GPU(duration=15)` — a tight
   realistic ceiling for queue priority.

Breaking any of these three typically shows up as CUDA init errors or very
slow first requests. See repo root `docs/ARCHITECTURE.md`.

## Before training a custom model

`app.py` defaults to `Qwen/Qwen2.5-3B-Instruct`, a stock instruct model.
Swap `MODEL_ID` for your DPO-aligned model from `training/train_dpo.py`
once it exists.

## Limitations

- No crisis detection — see repo root `docs/ROADMAP.md` Phase 1.5.
- `enforce_declarative_lock()` is a post-hoc regex filter on the model's
  output, not a generation-time constraint. It can produce an empty or
  truncated response if every sentence the model generated happened to be
  a question — the function falls back to the raw text in that case, which
  means the `?`-free guarantee is not absolute. For a hard guarantee, use a
  GBNF grammar constraint instead (see `arch3_neuroact_solitude/` for the
  pattern — `llama-cpp-python` supports this; the `transformers`-based
  pipeline here does not have an equivalent built in without extra work).
