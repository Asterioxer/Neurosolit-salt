import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";
import { assessRisk, CRISIS_MESSAGE, WATCH_NOTE } from "./crisis.js";

// Preferred local models: 1.5B provides far better instruction following than 0.5B.
const PRIMARY_MODEL_ID = "onnx-community/Qwen2.5-1.5B-Instruct";
const FALLBACK_MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";
const CUSTOM_MODEL_ID = null; // e.g. "your-username/nd-solitude-qwen-onnx"
let ACTIVE_MODEL_ID = CUSTOM_MODEL_ID || PRIMARY_MODEL_ID;

function enforceDeclarativeLock(rawText) {
  if (!rawText) return "";
  const sentences = rawText.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [rawText];
  const kept = [];
  for (const s of sentences) {
    const trimmed = s.trim();
    if (trimmed.endsWith("?")) continue;
    if (/^(why|how|what|when|where|who|could you|can you|would you|should you)\b/i.test(trimmed)) continue;
    kept.push(trimmed);
  }
  return kept.join(" ");
}

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
      try {
        this.generator = await pipeline("text-generation", ACTIVE_MODEL_ID, {
          device: this.device,
          dtype: CUSTOM_MODEL_ID ? "q8" : this.device === "webgpu" ? "q4f16" : "q8",
          progress_callback: onProgress,
        });
      } catch (err) {
        // Fallback to smaller 0.5B model if 1.5B fails memory or network allocation
        if (ACTIVE_MODEL_ID !== FALLBACK_MODEL_ID) {
          console.warn(`Failed loading ${ACTIVE_MODEL_ID}, falling back to ${FALLBACK_MODEL_ID}:`, err);
          ACTIVE_MODEL_ID = FALLBACK_MODEL_ID;
          this.generator = await pipeline("text-generation", ACTIVE_MODEL_ID, {
            device: this.device,
            dtype: this.device === "webgpu" ? "q4f16" : "q8",
            progress_callback: onProgress,
          });
        } else {
          throw err;
        }
      }
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
  solitude:
    "The assistant reflects quietly on solitude, silence, and recovery. " +
    "Never give advice, never ask questions, and never use commands. " +
    "Offer gentle, grounded declarative observations about the stillness of the space.",
};

self.addEventListener("message", async (e) => {
  const { type, text, mode } = e.data;

  if (type === "INIT") {
    try {
      await LocalSanctuaryEngine.getGenerator((info) => {
        self.postMessage({ status: "progress", info });
      });
      self.postMessage({ status: "ready", device: LocalSanctuaryEngine.device, model: ACTIVE_MODEL_ID });
    } catch (err) {
      self.postMessage({ status: "error", message: err.message });
    }
    return;
  }

  if (type === "REFLECT") {
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
        { role: "system", content: PROMPTS[mode] || PROMPTS.solitude },
        { role: "user", content: text },
      ];

      let buffer = "";
      const streamer = new TextStreamer(gen.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (tok) => {
          buffer += tok;
          const clean = enforceDeclarativeLock(buffer);
          self.postMessage({ status: "stream", text: clean || buffer.replace(/\?/g, ".") });
        },
      });

      await gen(messages, {
        max_new_tokens: 120,
        temperature: 0.35,
        do_sample: true,
        top_p: 0.9,
        streamer,
      });

      const finalClean = enforceDeclarativeLock(buffer) || buffer.replace(/\?/g, ".");
      if (risk === "watch") {
        self.postMessage({ status: "stream", text: finalClean + "\n\n" + WATCH_NOTE });
      } else {
        self.postMessage({ status: "stream", text: finalClean });
      }
      self.postMessage({ status: "done" });
    } catch (err) {
      self.postMessage({ status: "error", message: err.message });
    }
  }
});
