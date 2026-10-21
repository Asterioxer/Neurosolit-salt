---
title: Liminal Space Sanctuary
emoji: 🌑
colorFrom: gray
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# Liminal-Space: Client-Side WebGPU Epistolary Sanctuary

Reflection space where inference runs in the browser (WebGPU, WASM fallback) inside a Web Worker.
Text you type is never sent to a server. The page does fetch the transformers.js library from
jsDelivr and the model weights from the Hugging Face CDN when you press "Load model"; that is
code and weights only, never your text.

Built for schizoid personality structure and nocturnal hikikomori-adjacent users for whom the
act of disclosure is itself the primary threat.

## Run locally

Browsers block module workers on `file://`, so serve the folder:

```bash
python -m http.server 8000 --directory arch1_liminal_space
# open http://localhost:8000
```

## Deploy

From the repo root: `python scripts/deploy.py --hf-username YOU` (static Space, free).

## Model

`worker.js` defaults to `onnx-community/Qwen2.5-0.5B-Instruct`, which transformers.js 3.3.3
supports. After running the training pipeline, deploy with `--onnx-repo YOU/nd-solitude-qwen-onnx`
and `deploy.py` sets `CUSTOM_MODEL_ID` in `worker.js` for you.

## Safety behavior

`crisis.js` gates on the main thread before anything reaches the worker, and again inside the
worker. Text matching explicit suicide/self-harm patterns shows a fixed list of support lines and
is never passed to the model. It is a keyword screen, not a clinical assessment, and needs
clinician review before public use. It mirrors `arch2_declarative_ally/safety.py`.

## Limitations

- Model quality is capped by what fits in browser memory (0.5B default).
- The `?` to `.` replacement is a syntax filter, not a content safety check.
