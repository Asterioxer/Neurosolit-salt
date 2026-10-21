# Roadmap

Phases 0–7 are the engineering build-out. Phase 1.5 and the "Hardening"
section at the bottom are not yet designed — they're the clinical-safety
work that should happen before any public deployment, and are tracked as
open issues rather than code, because they need human (ideally clinical)
judgment, not just engineering.

## Phase 0 — Environment setup
- [x] Install `huggingface_hub[cli]`, authenticate with a Write token
- [x] Install Git LFS
- **Output:** authenticated local environment, ready to create repos/Spaces

## Phase 1 — Dataset curation
- [x] Run `dataset/build_dpo_dataset.py` against an open counseling corpus
- [x] Review every generated `chosen`/`rejected` pair by hand before training —
      the heuristic converter in the script is crude and will produce some
      nonsensical or clinically wrong "chosen" responses
- [x] Push the reviewed dataset to the Hub
- **Output:** a DPO preference dataset repo on HF Hub

## Phase 1.5 — Crisis detection layer (Implemented with Normalized Keyword & Support Screen)
- [x] Define crisis signal vs. schizoid-withdrawal signal (hard crisis screen steps aside before model inference)
- [x] Hard safety gate: routes to Tele-MANAS (14416), 988, Samaritans (116 123) with zero-question low-demand copy
- [x] Obfuscation/spacing/zero-width normalization across all 3 Space mirrors (`safety.py` and `crisis.js`)
- [ ] Clinical review by external licensed practitioner before unrestricted public promotion

## Phase 2 — Model training (QLoRA SFT + span-aware DPO)
- [x] Run SFT/DPO pipeline (`training/train_sft.py`, `run_pipeline.py`)
- [x] Automated evaluation gates passed (0 `?`, <=1 soft-imperative, word bounds, trigram diversity)
- [x] Merged model weights pushed to HF Hub (`Scadiva/nd-solitude-qwen-1.5b-merged`)

## Phase 3 — Format conversion
- [x] `training/export_onnx.py` → quantized ONNX for Architecture 1 (browser)
- [x] `training/export_gguf.sh` → Q4_K_M GGUF for Architecture 3 (CPU Space)
- **Output:** model formats ready on HF Hub

## Phase 4 — Deploy Architecture 1 (Liminal-Space)
- [x] Deploy `arch1_liminal_space/` as static Space on HF (`Scadiva/liminal-space`)
- [x] Zero server inference via client-side WebGPU / WASM
- **Output:** live Space, zero server inference

## Phase 5 — Deploy Architecture 2 (Declarative-Ally)
- [x] Deploy `arch2_declarative_ally/` on HF (`Scadiva/declarative-ally`)
- [x] ZeroGPU with CPU fallback and fine-tuned model variable `NEURO_MODEL_ID`
- **Output:** live ZeroGPU Space

## Phase 6 — Deploy Architecture 3 (NeuroACT-Solitude)
- [x] Deploy `arch3_neuroact_solitude/` on HF (`Scadiva/neuroact-solitude`)
- [x] RAG clinical guidance, circuit breakers, and fine-tuned model
- **Output:** live Space

## Phase 7 — Verification
- [x] Unit tests passing (`pytest` / `tests/test_safety.py`)
- [x] Multi-platform safety pattern synchronization passing (`scripts/check_sync.py`)
- [x] Live automated smoke tests passing (`scripts/smoke_test.py`)

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
