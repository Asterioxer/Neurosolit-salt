# Architecture rationale

Three Spaces, each matched to a different clinical threat profile and a
different free HF compute tier.

## Why three separate apps instead of one

The populations these target have structurally different core threats:

- **Schizoid / SZPD**: the threat is *data leaving the device at all* —
  engulfment anxiety is triggered by disclosure itself, not by what's done
  with the disclosure. Needs zero server round-trip.
- **PDA autism**: the threat is *perceived demand* — imperatives, questions,
  even implicit ones. Needs fast, accurate rewriting of incoming text, which
  benefits from a larger model than a phone/laptop can run locally.
- **Chronic withdrawal / hikikomori**: the threat is *unpredictability and
  short-session friction* — needs session continuity and retrieval over a
  stable knowledge base, not raw model capability.

One architecture optimized for all three would compromise on all three.

## Architecture 1 — Liminal-Space (Static Space, client-side WebGPU)

- **Compute tier:** `--space-sdk static`, $0/mo, zero server inference
- **Model:** ONNX, runs in-browser via `@huggingface/transformers` + WebGPU
  (WASM fallback if no WebGPU)
- **Why this shape:** the only way to structurally guarantee "nothing you
  type leaves this device" is to not have a server in the inference path.
  A Web Worker keeps token generation off the UI thread.
- **Trade-off:** limited to small models (≤3B realistically) that fit in
  browser memory and run at usable speed on consumer GPUs/CPUs.

## Architecture 2 — Declarative-Ally (ZeroGPU Space)

- **Compute tier:** `--space-sdk gradio --flavor zero-a10g`, $0/mo, dynamic
  GPU (A10G-class) only during the `@spaces.GPU`-decorated call
- **Model:** a larger instruct model (3B+) for higher-quality rewriting
- **Why this shape:** translating arbitrary incoming text into BERMED
  declarative form is a harder generation task than the other two apps do —
  it benefits from more capability than a 0.5–1.5B model offers, and
  ZeroGPU gives that for free in short bursts.
- **Three non-negotiable ZeroGPU rules** (see `app.py` comments): `import
  spaces` before `import torch`; load model at module scope and move to
  `.to("cuda")` eagerly; decorate the inference function with
  `@spaces.GPU(duration=...)`.

## Architecture 3 — NeuroACT-Solitude (CPU-Basic Space)

- **Compute tier:** `--space-sdk gradio --flavor cpu-basic`, $0/mo, 2 vCPU /
  16GB RAM, always-on (no cold-start GPU queue)
- **Model:** Q4_K_M GGUF via `llama-cpp-python`, plus a small embedding
  model (`BAAI/bge-small-en-v1.5`) for retrieval over a fixed clinical
  knowledge base
- **Why this shape:** long-term, low-urgency conversation doesn't need GPU
  burst capacity — it needs to be *always available without a queue*, which
  CPU-Basic provides and ZeroGPU doesn't guarantee under load.
- **Circuit breaker:** a regex/keyword pre-check (`check_circuit_breaker`)
  flags likely engulfment-panic or verbal-shutdown language before
  generation, and overrides the system prompt accordingly. This is a crude
  heuristic, not a safety system — see `docs/ROADMAP.md` Phase 1.5.

## Shared pattern: guardrail smoke testing

All three apps should be checked post-deploy for whether they actually
avoid the syntactic patterns (`?`, imperative-verb openers) the clinical
framing asks them to avoid. For the two server-side apps this can be
automated with `gradio_client`:

```python
from gradio_client import Client
client = Client("<user>/declarative-ally")
result = client.predict("...", "Standard Low-Demand", api_name="/neutralize_demands")
assert "?" not in result
```

This checks syntax, not clinical appropriateness — a response with no
question marks can still be a bad response. Treat it as a minimum bar, not
a quality gate.
