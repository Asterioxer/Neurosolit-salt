import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

// Default: stock instruct model so the Space works before any custom
// training is done. Swap for your own ONNX export (see training/export_onnx.py)
// once you have one.
const MODEL_ID = "HuggingFaceTB/SmolLM3-3B-ONNX"; // or "onnx-community/Qwen2.5-0.5B-Instruct" for a lighter load

class LocalSanctuaryEngine {
  static generator = null;

  static async getGenerator(onProgress) {
    if (!this.generator) {
      const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;
      this.generator = await pipeline("text-generation", MODEL_ID, {
        device: hasWebGPU ? "webgpu" : "wasm",
        dtype: hasWebGPU ? "q4f16" : "q4",
        progress_callback: onProgress,
      });
    }
    return this.generator;
  }
}

const PROMPTS = {
  analytical:
    "The assistant is an impersonal, objective analytical mirror running locally inside an offline vault. " +
    "Never ask questions. Never use imperative verbs or suggest socializing. " +
    "Provide brief, third-person structural observations about the concepts or patterns written, " +
    "respecting total emotional distance and self-sufficiency.",
  declarative:
    "The assistant uses strictly low-demand declarative statements ('It appears...', 'One pattern here is...'). " +
    "Ask zero questions and require zero reply. Validate the peace of solitude and staying put.",
};

self.addEventListener("message", async (e) => {
  const { type, text, mode } = e.data;

  if (type === "INIT") {
    try {
      await LocalSanctuaryEngine.getGenerator((info) => {
        self.postMessage({ status: "progress", info });
      });
      self.postMessage({ status: "ready" });
    } catch (err) {
      self.postMessage({ status: "error", message: err.message });
    }
    return;
  }

  if (type === "REFLECT") {
    try {
      const gen = await LocalSanctuaryEngine.getGenerator();
      const messages = [
        { role: "system", content: PROMPTS[mode] || PROMPTS.analytical },
        { role: "user", content: text },
      ];

      let buffer = "";
      const streamer = new TextStreamer(gen.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (tok) => {
          buffer += tok;
          // Syntax-level guardrail only -- not a content safety check.
          self.postMessage({ status: "stream", text: buffer.replace(/\?/g, ".") });
        },
      });

      await gen(messages, {
        max_new_tokens: 130,
        temperature: 0.35,
        do_sample: true,
        top_p: 0.9,
        streamer,
      });

      self.postMessage({ status: "done" });
    } catch (err) {
      self.postMessage({ status: "error", message: err.message });
    }
  }
});
