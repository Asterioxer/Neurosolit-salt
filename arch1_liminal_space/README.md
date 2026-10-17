---
title: Liminal Space Sanctuary
emoji: 🌑
colorFrom: gray
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# Liminal-Space: Zero-Telemetry Local WebGPU Epistolary Sanctuary

100% client-side, offline-capable reflection space. No text typed here is
ever sent to a server — inference runs entirely in the browser via WebGPU
(or WASM as a fallback) inside a Web Worker.

Built for schizoid personality structure (SZPD) and nocturnal
hikikomori-adjacent users for whom the act of disclosure itself — not what
happens to the disclosed content — is the primary threat.

## Deploy

```bash
hf repos create <your-username>/liminal-space \
  --type space --space-sdk static --public --exist-ok

git clone https://huggingface.co/spaces/<your-username>/liminal-space
cd liminal-space
cp /path/to/this/directory/* .
git add . && git commit -m "Deploy Liminal-Space" && git push
```

## Before training a custom model

`worker.js` defaults to `HuggingFaceTB/SmolLM3-3B-ONNX`, a stock instruct
model — this lets you deploy and test the UI immediately without waiting on
`training/`. Once you have a DPO-aligned model exported via
`training/export_onnx.py`, update `MODEL_ID` in `worker.js`.

## Limitations

- No crisis detection. See repo root `docs/ROADMAP.md` Phase 1.5.
- Model quality is capped by what fits comfortably in browser memory
  (realistically ≤3B parameters).
- The `.replace(/\?/g, ".")` guardrail in `worker.js` is a syntax filter on
  the streamed output, not a content safety check.
