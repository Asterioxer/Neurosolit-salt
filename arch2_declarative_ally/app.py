try:
    import spaces  # ZeroGPU rule 1: import before torch.
except ImportError:
    # No-op mock for local development and non-ZeroGPU environments
    class _SpacesMock:
        @staticmethod
        def GPU(duration=60):
            def decorator(fn):
                return fn
            return decorator
    spaces = _SpacesMock()

import os
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety import CRISIS_MESSAGE, WATCH_NOTE, assess_risk

# ---------------------------------------------------------------------------
# Hardware-adaptive model loading.
#   ZeroGPU / any CUDA box -> 3B model in fp16 on the GPU
#   CPU-only (cpu-basic Space, a laptop) -> 1.5B model in fp32 on the CPU
# Override with env var NEURO_MODEL_ID (e.g. your DPO-merged model from
# training/train_dpo.py). deploy.py can set this for you with --merged-repo.
# ---------------------------------------------------------------------------
USE_CUDA = bool(os.getenv("SPACES_ZERO_GPU")) or torch.cuda.is_available()
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct" if USE_CUDA else "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ID = os.getenv("NEURO_MODEL_ID") or DEFAULT_MODEL
DEVICE = "cuda" if USE_CUDA else "cpu"

try:
    from peft import PeftConfig, PeftModel
    peft_config = PeftConfig.from_pretrained(MODEL_ID)
    base_id = peft_config.base_model_name_or_path
    print(f"[declarative-ally] Loading base model {base_id} for adapter {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if USE_CUDA:
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16).to("cuda")
    else:
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, MODEL_ID)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if USE_CUDA:
        # ZeroGPU rule 2: load at module scope, move to cuda eagerly.
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print(f"[declarative-ally] model={MODEL_ID} device={DEVICE}")

BERMED_SYSTEM_PROMPT = (
    "Act as a clinical specialist in Linda K. Murphy's BERMED Declarative Language Framework "
    "for Pervasive Drive for Autonomy (PDA) and Autistic Inertia. "
    "Rewrite the incoming text into a 100% low-demand declarative observation. "
    "Strict constraints: "
    "1. Use ZERO imperative verbs (never tell the reader what to do). "
    "2. Use ZERO questions and never output a '?' character. "
    "3. Frame thoughts using observational prefixes ('I notice...', 'It looks like...', 'One option is...'). "
    "4. Explicitly affirm that pausing, staying still, or saying 'no' is completely safe."
)


def enforce_declarative_lock(text: str) -> str:
    """
    Post-hoc syntactic guardrail: drops sentences ending in '?' and neutralizes
    '!' / stray '?'. If every generated sentence was an interrogative, it falls back
    to a calm neutral observation rather than converting questions into dotted sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    safe = [s for s in sentences if not s.endswith("?")]
    cleaned = " ".join(safe) if safe else "This space remains neutral and unhurried. No response is needed."
    return cleaned.replace("?", ".").replace("!", ".")


def _generate(raw_demand_text: str, bandwidth_mode: str) -> str:
    max_new_tokens = 55 if bandwidth_mode == "Acute Overload (2 Sentences Max)" else 150
    messages = [
        {"role": "system", "content": BERMED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Bandwidth State: {bandwidth_mode}\nIncoming High-Demand Text:\n{raw_demand_text}",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.85,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = output_ids[0][inputs.input_ids.shape[-1]:]
    return enforce_declarative_lock(tokenizer.decode(gen_tokens, skip_special_tokens=True))


# ZeroGPU rule 3: tight duration on the GPU-decorated function (no-op elsewhere).
@spaces.GPU(duration=20)
def _generate_gpu(raw_demand_text: str, bandwidth_mode: str) -> str:
    return _generate(raw_demand_text, bandwidth_mode)


def neutralize_demands(raw_demand_text: str, bandwidth_mode: str) -> str:
    if not raw_demand_text.strip():
        return "Environment is quiet. Zero demands active."

    # Hard crisis gate: runs before any model call, no generation involved.
    risk = assess_risk(raw_demand_text)
    if risk == "crisis":
        return CRISIS_MESSAGE

    out = _generate_gpu(raw_demand_text, bandwidth_mode) if USE_CUDA else _generate(raw_demand_text, bandwidth_mode)
    if risk == "watch":
        out = out + "\n\n" + WATCH_NOTE
    return out


def get_shutdown_card(card_type: str) -> str:
    """Zero-compute, pre-written boundary scripts for acute shutdown use."""
    cards = {
        "Request Sequential Departure (Need Solitude)": (
            "Sensory and spoken language processing are currently depleted. Nothing is wrong, "
            "and no questions or assistance are needed. Allowing the room to empty out into "
            "quiet solitude is the safest way to help recovery right now."
        ),
        "Autistic Inertia Freeze (Stay Put)": (
            "Experiencing autistic inertia and motor freeze right now. Please avoid prompting "
            "movement, eye contact, or verbal replies. Remaining stationary in quiet will allow "
            "the nervous system to reset naturally."
        ),
        "Post-Social Decompression": (
            "Entering low-demand solitary decompression mode for the rest of the day. "
            "Messages and expectations are paused until bandwidth returns."
        ),
        "Asynchronous / Workplace Low-Bandwidth Status": (
            "Currently operating in asynchronous, low-bandwidth mode to manage cognitive overload. "
            "Urgent direct calls or rapid replies are paused. Written notes will be processed "
            "as energy permits, with zero expectation of immediate response."
        ),
        "Sensory / Auditory Overload Space Request": (
            "Auditory processing is currently overwhelmed. Spoken requests cannot be parsed safely. "
            "Please allow quiet physical space and reduce ambient noise. No reply is expected."
        ),
    }
    return cards.get(card_type, "")


with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Declarative-Ally: BERMED Demand De-Escalator & Shutdown Proxy")
    gr.Markdown(
        "_Unvalidated prototype, not a therapist replacement. Text that sounds like thoughts of "
        "ending life or self-harm is not rewritten; a fixed list of support lines is shown instead._"
    )

    with gr.Tab("BERMED Demand De-Escalator"):
        demand_in = gr.Textbox(
            label="Paste High-Demand Email, Instruction, or Internal Checklist",
            lines=4,
            placeholder="e.g., You need to finish this report by 5 PM and call the clinic immediately!",
        )
        bw_mode = gr.Radio(
            choices=["Standard Low-Demand", "Acute Overload (2 Sentences Max)"],
            value="Standard Low-Demand",
            label="Current Nervous System Bandwidth",
        )
        convert_btn = gr.Button("Neutralize into BERMED Declarative Observation", variant="primary")
        declarative_out = gr.Textbox(label="Zero-Demand Declarative Output", lines=4, show_copy_button=True)
        convert_btn.click(
            fn=neutralize_demands,
            inputs=[demand_in, bw_mode],
            outputs=declarative_out,
            api_name="neutralize_demands",
        )

        gr.Examples(
            examples=[
                [
                    "You must attend the team social event tonight and tell everyone about your progress! Why are you always avoiding these meetings?",
                    "Standard Low-Demand",
                ],
                [
                    "You have to finish this report by 5 PM, submit your taxes, and call the clinic right away!",
                    "Acute Overload (2 Sentences Max)",
                ],
                [
                    "Can you please review this entire document immediately and send me all your feedback?",
                    "Acute Overload (2 Sentences Max)",
                ],
            ],
            inputs=[demand_in, bw_mode],
            outputs=declarative_out,
            fn=neutralize_demands,
            cache_examples=False,
        )

    with gr.Tab("Nonverbal Shutdown Proxy Cards"):
        card_choices = [
            "Request Sequential Departure (Need Solitude)",
            "Autistic Inertia Freeze (Stay Put)",
            "Post-Social Decompression",
            "Asynchronous / Workplace Low-Bandwidth Status",
            "Sensory / Auditory Overload Space Request",
        ]
        card_selector = gr.Radio(
            choices=card_choices,
            value="Request Sequential Departure (Need Solitude)",
            label="Select Zero-Typing Boundary Script",
        )
        card_out = gr.Textbox(
            label="Display or Copy Script",
            value=get_shutdown_card("Request Sequential Departure (Need Solitude)"),
            lines=3,
            show_copy_button=True,
        )
        card_selector.change(fn=get_shutdown_card, inputs=card_selector, outputs=card_out, api_name=False)

if __name__ == "__main__":
    demo.queue().launch()
