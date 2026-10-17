# Neurosolit-salt

Low-demand, solitude-affirming conversational AI tools for neurodivergent people
(schizoid personality structure, PDA autism, autistic shutdown/burnout, and
hikikomori-adjacent withdrawal) — built to deploy entirely on free Hugging Face
compute tiers.

This repo is a **scaffold**: three independent Space apps, a dataset-building
script, and a training pipeline. Nothing here is trained or deployed yet.
See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the phase-by-phase build order and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the three apps differ and
why.

## ⚠️ Read this before building

This project touches real mental-health crisis territory — several of the
seed training examples in `dataset/` contain language like "the urge to
delete this account and disappear," which sits close to self-harm ideation
phrasing even though the clinical intent (schizoid withdrawal) is distinct
from it. Before training or deploying anything from this repo:

- **This is not a crisis-response system.** None of the three architectures
  described here include crisis detection, escalation, or human handoff. If
  you deploy any of these publicly, you need to add that layer first —
  see `docs/ROADMAP.md` Phase 1.5 (not yet built) and the open issue for it.
- **The DPO alignment approach suppresses *all* questions and imperatives**,
  including ones that would be clinically necessary in an actual crisis
  (e.g. "are you safe right now"). A GBNF grammar that blocks `?` characters
  is a syntax filter, not a safety system. Treat every architecture here as
  a prototype that needs a human-reviewed safety layer before anyone
  vulnerable uses it unsupervised.
- **No architecture here has been clinically validated.** The framing
  (BERMED declarative language, schizoid-process theory, NeuroACT) is drawn
  from published clinical literature, but the alignment pipeline itself is
  original engineering work, not a validated intervention.

## What's in here

| Directory | Purpose |
|---|---|
| `dataset/` | Script to build the DPO preference dataset from open counseling corpora + seed clinical pairs |
| `training/` | QLoRA SFT + span-aware DPO training script, ONNX/GGUF export scripts |
| `arch1_liminal_space/` | Static HF Space — 100% client-side WebGPU, zero telemetry (for SZPD / engulfment-sensitive users) |
| `arch2_declarative_ally/` | ZeroGPU HF Space — real-time demand-to-declarative translator (for PDA / shutdown) |
| `arch3_neuroact_solitude/` | CPU-Basic HF Space — stateful GGUF + RAG companion (for chronic withdrawal / hikikomori) |
| `docs/` | Roadmap, architecture rationale, clinical framework notes |

## Quick start

Nothing deploys out of the box — you need your own Hugging Face account and
a Write token. Start with `docs/ROADMAP.md` Phase 0.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login --token hf_YOUR_WRITE_TOKEN
```

Then follow the roadmap phase by phase. Architecture 1 (`arch1_liminal_space/`)
is the lowest-risk starting point — it has no server component and no
clinical-dataset dependency to get working, since it can run on an
off-the-shelf instruction model before any DPO alignment exists.

## License

Code in this repo: your choice (add a LICENSE file before publishing).
Note that `Amod/mental_health_counseling_conversations` (used optionally in
`dataset/build_dpo_dataset.py`) is RAIL-D licensed — free for non-commercial
research filtering, commercial use requires a $100 donation to the CCC
Foundation helpline. See the script's docstring.
