# Neurosolit-salt

Low-demand, solitude-affirming conversational tools for neurodivergent people (schizoid
personality structure, PDA autism, autistic shutdown/burnout, hikikomori-adjacent withdrawal),
built to deploy on free Hugging Face tiers.

## Live Deployments

🌐 **Interactive Web Hub (GitHub Pages):** [https://asterioxer.github.io/Neurosolit-salt/](https://asterioxer.github.io/Neurosolit-salt/)

### Hugging Face Spaces & Models

| App | Live URL | Compute & Hardware |
|---|---|---|
| `arch1_liminal_space` | [Scadiva/liminal-space](https://huggingface.co/spaces/Scadiva/liminal-space) | Static Space (100% Client-Side WebGPU / WASM) |
| `arch2_declarative_ally` | [Scadiva/declarative-ally](https://huggingface.co/spaces/Scadiva/declarative-ally) | ZeroGPU (A10G) / CPU Fallback |
| `arch3_neuroact_solitude` | [Scadiva/neuroact-solitude](https://huggingface.co/spaces/Scadiva/neuroact-solitude) | ZeroGPU (A10G) / CPU Fallback |

Fine-tuned weights:
- **7B Model (Current ZeroGPU Active • High Intelligence):** [`Scadiva/nd-solitude-7b-adapter`](https://huggingface.co/Scadiva/nd-solitude-7b-adapter) (Qwen2.5-7B Instruct + QLoRA low-demand declarative fine-tuning on 142 curated pairs)
- **3B Model (Balanced Fallback):** [`Scadiva/nd-solitude-3b-merged`](https://huggingface.co/Scadiva/nd-solitude-3b-merged) (Qwen2.5-3B Instruct SFT)
- **1.5B Model (Lightweight Fallback):** [`Scadiva/nd-solitude-qwen-1.5b-merged`](https://huggingface.co/Scadiva/nd-solitude-qwen-1.5b-merged) (Qwen2.5-1.5B Instruct + DPO)

### Interactive Terminal Testing
You can chat with or test the deployed spaces directly from your terminal:
```bash
python scripts/chat_live.py --app arch3    # Chat with NeuroACT-Solitude companion
python scripts/chat_live.py --app arch2    # Test Declarative-Ally demand neutralization
```

## Quick start

```bash
pip install -r requirements-dev.txt
hf auth login                       # once, with a WRITE token
python run_all.py --hf-username YOU
```

That runs the safety tests, builds the dataset, deploys all three Spaces with stock models, waits
for the builds, and smoke-tests them. Each app also runs locally on its own:

| App | Local run | Free tier |
|---|---|---|
| `arch1_liminal_space` | `python -m http.server 8000 --directory arch1_liminal_space` | static Space |
| `arch2_declarative_ally` | `cd arch2_declarative_ally && pip install -r requirements.txt && python app.py` | ZeroGPU if your account can host it, else CPU-Basic (auto-adapts) |
| `arch3_neuroact_solitude` | `cd arch3_neuroact_solitude && pip install -r requirements.txt && python app.py` | CPU-Basic |

## Custom model (optional, needs a free Colab GPU)

Open `training/colab_pipeline.ipynb` in Colab. It builds the dataset, trains the QLoRA DPO model,
exports ONNX and GGUF, and prints the final deploy command, which looks like:

```bash
python scripts/deploy.py --hf-username YOU --onnx-repo YOU/nd-solitude-qwen-onnx \
  --merged-repo YOU/nd-solitude-qwen-1.5b-merged --gguf-repo YOU/nd-solitude-qwen-gguf
```

## Safety layer (read this)

All three apps run a hard crisis gate before any model call (`crisis.js` / `safety.py`). Text
that matches explicit suicide or self-harm patterns is never sent to the model; a fixed message
with support lines (Tele-MANAS 14416, 988, Samaritans 116 123, findahelpline.com) is shown.
Ambiguous distress phrases get a one-time support note. The dataset builder also drops every
training row containing that language, so the model is never taught to answer it with low-demand
validation.

This is a keyword screen, not a clinical assessment. It misses indirect phrasing. The seed
training pairs and the knowledge base were not written or reviewed by a clinician, and nothing
here is validated as an intervention. Have someone with clinical training review `safety.py`, the
seed pairs, and `knowledge_base.json` before promoting any Space beyond personal use. The
hardening backlog in `docs/ROADMAP.md` still applies.

## Layout

| Path | Purpose |
|---|---|
| `run_all.py` | one-command tests, dataset, deploy, smoke |
| `scripts/deploy.py` | Python deploy (Windows-friendly); replaces the old bash script |
| `scripts/check_sync.py`, `tests/` | keep the three safety copies identical, test the gate |
| `dataset/` | DPO pair builder with crisis filtering and auto-filters |
| `training/` | DPO training, ONNX/GGUF export, Colab notebook |
| `docs/` | architecture rationale and roadmap |

`scripts/deploy_all_free_spaces.sh` is the old bash deploy path and is superseded by `deploy.py`.

## License

This project is licensed under the MIT License with non-clinical research prototype notices (see [LICENSE](LICENSE)). The optional `Amod/mental_health_counseling_conversations`
corpus is RAIL-D licensed; see the docstring in `dataset/build_dpo_dataset.py`.
