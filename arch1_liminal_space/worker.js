import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";
import { assessRisk, CRISIS_MESSAGE, WATCH_NOTE } from "./crisis.js";

// Default: a small stock instruct model that transformers.js 3.3.3 supports,
// so the Space works before any custom training. After running the pipeline
// (training/export_onnx.py) set CUSTOM_MODEL_ID to your ONNX repo; deploy.py
// can do this for you with --onnx-repo.
const DEFAULT_MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";
const CUSTOM_MODEL_ID = null; // e.g. "your-username/nd-solitude-qwen-onnx"
const MODEL_ID = CUSTOM_MODEL_ID || DEFAULT_MODEL_ID;

class LocalSanctuaryEngine {
  static generator = null;
  static device = null;

  static async pickDevice() {
    try {
      if (typeof navigator !== "undefined" && navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) return "webgpu";
      }
    } catch (_) {
      /* fall through to wasm */
    }
    return "wasm";
  }

  static async getGenerator(onProgress) {
    if (!this.generator) {
      this.device = await this.pickDevice();
      this.generator = await pipeline("text-generation", MODEL_ID, {
        device: this.device,
        dtype: CUSTOM_MODEL_ID ? "q8" : this.device === "webgpu" ? "q4f16" : "q8",
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
      self.postMessage({ status: "ready", device: LocalSanctuaryEngine.device, model: MODEL_ID });
    } catch (err) {
      self.postMessage({ status: "error", message: err.message });
    }
    return;
  }

  if (type === "REFLECT") {
    // Second line of defense -- the page also gates before posting here.
    const risk = assessRisk(text);
    if (risk === "crisis") {
      self.postMessage({ status: "crisis", text: CRISIS_MESSAGE });
      return;
    }

    try {
      const gen = await LocalSanctuaryEngine.getGenerator((info) => {
        self.postMessage({ status: "progress", info });
      });
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

      if (risk === "watch") {
        self.postMessage({ status: "stream", text: buffer.replace(/\?/g, ".") + "\n\n" + WATCH_NOTE });
      }
      self.postMessage({ status: "done" });
    } catch (err) {
      self.postMessage({ status: "error", message: err.message });
    }
  }
});
