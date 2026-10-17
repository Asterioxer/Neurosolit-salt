# Roadmap

Phases 0–7 are the engineering build-out. Phase 1.5 and the "Hardening"
section at the bottom are not yet designed — they're the clinical-safety
work that should happen before any public deployment, and are tracked as
open issues rather than code, because they need human (ideally clinical)
judgment, not just engineering.

## Phase 0 — Environment setup
- [ ] Install `huggingface_hub[cli]`, authenticate with a Write token
- [ ] Install Git LFS
- **Output:** authenticated local environment, ready to create repos/Spaces

## Phase 1 — Dataset curation
- [ ] Run `dataset/build_dpo_dataset.py` against an open counseling corpus
- [ ] Review every generated `chosen`/`rejected` pair by hand before training —
      the heuristic converter in the script is crude and will produce some
      nonsensical or clinically wrong "chosen" responses
- [ ] Push the reviewed dataset to the Hub
- **Output:** a DPO preference dataset repo on HF Hub

## Phase 1.5 — Crisis detection layer (NOT YET DESIGNED — open issue)
- [ ] Define what counts as a crisis signal vs. a schizoid-withdrawal signal
      (these can look similar in text — see README warning)
- [ ] Decide what the system does on detection: refuse to continue in
      low-demand mode, surface a crisis resource, or something else
- [ ] This almost certainly needs input from someone with clinical training,
      not just engineering judgment
- **This phase blocks public deployment of any architecture below.**

## Phase 2 — Model training (QLoRA SFT + span-aware DPO)
- [ ] Run `training/train_dpo.py` on a free Colab/Kaggle GPU
- [ ] Manually evaluate the merged model against held-out prompts —
      check it doesn't produce the `?`/imperative violations it was trained
      against, AND check it doesn't degenerate into refusing to engage at all
- **Output:** merged model weights pushed to HF Hub

## Phase 3 — Format conversion
- [ ] `training/export_onnx.py` → quantized ONNX for Architecture 1 (browser)
- [ ] `training/export_gguf.sh` → Q4_K_M GGUF for Architecture 3 (CPU Space)
- **Output:** two additional model repos on HF Hub

## Phase 4 — Deploy Architecture 1 (Liminal-Space)
- [ ] `hf repos create <user>/liminal-space --space-sdk static`
- [ ] Push `arch1_liminal_space/` contents
- [ ] Can be deployed with a stock ONNX model (e.g. `onnx-community/Qwen2.5-0.5B-Instruct`)
      *before* Phase 2–3 are done, as a UI/UX testbed
- **Output:** live Space, zero server inference

## Phase 5 — Deploy Architecture 2 (Declarative-Ally)
- [ ] `hf repos create <user>/declarative-ally --space-sdk gradio --flavor zero-a10g`
- [ ] Push `arch2_declarative_ally/` contents
- **Output:** live ZeroGPU Space

## Phase 6 — Deploy Architecture 3 (NeuroACT-Solitude)
- [ ] `hf repos create <user>/neuroact-solitude --space-sdk gradio --flavor cpu-basic`
- [ ] Push `arch3_neuroact_solitude/` contents
- **Output:** live CPU-Basic Space

## Phase 7 — Verification
- [ ] `hf spaces logs <user>/<space> --build --follow` on each Space
- [ ] Run smoke tests (see `docs/ARCHITECTURE.md` for the guardrail-check pattern)

---

## Hardening backlog (open issues, not yet scheduled)
- [ ] Crisis detection / escalation (blocks public launch — see Phase 1.5)
- [ ] Rate limiting / abuse prevention on the ZeroGPU and CPU Spaces
- [ ] A way for a user to export and delete their own session state
      (Architecture 3 currently keeps conversation history in server memory
      per session — needs review for how long that persists)
- [ ] Independent review of the seed clinical pairs in `dataset/` by someone
      with relevant clinical background — they were written by synthesizing
      published literature, not by a clinician
- [ ] Decide on a disclosure/consent screen for each Space: users should know
      this is an unvalidated prototype, not a therapist replacement
