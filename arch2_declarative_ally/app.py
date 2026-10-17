import spaces  # RULE 1: must be imported before torch or transformers

import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Stock instruct model by default -- swap for your DPO-aligned model
# (training/train_dpo.py output) once you have one.
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# RULE 2: load eagerly at module scope and move to .to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to("cuda")
model.eval()

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
    Post-hoc syntactic guardrail: strips sentences ending in '?' and
    neutralizes exclamation marks. NOT a hard guarantee -- if every
    generated sentence happened to be a question, this falls back to the
    unmodified text. See README.md Limitations.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    safe_sentences = [s for s in sentences if not s.endswith("?")]
    cleaned = " ".join(safe_sentences) if safe_sentences else text
    return cleaned.replace("?", ".").replace("!", ".")


# RULE 3: tight duration estimate on the GPU-decorated function
@spaces.GPU(duration=15)
def neutralize_demands(raw_demand_text: str, bandwidth_mode: str) -> str:
    if not raw_demand_text.strip():
        return "Environment is quiet. Zero demands active."

    max_new_tokens = 55 if bandwidth_mode == "Acute Overload (2 Sentences Max)" else 150
    messages = [
        {"role": "system", "content": BERMED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Bandwidth State: {bandwidth_mode}\nIncoming High-Demand Text:\n{raw_demand_text}",
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

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
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return enforce_declarative_lock(decoded)


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
    }
    return cards.get(card_type, "")


with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Declarative-Ally: ZeroGPU BERMED Demand De-Escalator & Shutdown Proxy")
    gr.Markdown(
        "_This is an unvalidated prototype, not a therapist replacement. "
        "If you are in crisis, please reach out to a crisis line or someone you trust._"
    )

    with gr.Tab("BERMED Demand De-Escalator"):
        demand_in = gr.Textbox(
            label="Paste High-Demand Email, Instruction, or Internal Checklist",
            lines=4,
            placeholder=(
                "e.g., You need to finish this report by 5 PM and call the clinic "
                "immediately! Why haven't you started?"
            ),
        )
        bw_mode = gr.Radio(
            choices=["Standard Low-Demand", "Acute Overload (2 Sentences Max)"],
            value="Standard Low-Demand",
            label="Current Nervous System Bandwidth",
        )
        convert_btn = gr.Button("Neutralize into BERMED Declarative Observation")
        declarative_out = gr.Textbox(label="Zero-Demand Declarative Output", lines=4)
        convert_btn.click(fn=neutralize_demands, inputs=[demand_in, bw_mode], outputs=declarative_out)

    with gr.Tab("Nonverbal Shutdown Proxy Cards"):
        card_selector = gr.Radio(
            choices=[
                "Request Sequential Departure (Need Solitude)",
                "Autistic Inertia Freeze (Stay Put)",
                "Post-Social Decompression",
            ],
            value="Request Sequential Departure (Need Solitude)",
            label="Select Zero-Typing Boundary Script",
        )
        card_out = gr.Textbox(
            label="Display or Copy Script",
            value=get_shutdown_card("Request Sequential Departure (Need Solitude)"),
            lines=3,
        )
        card_selector.change(fn=get_shutdown_card, inputs=card_selector, outputs=card_out)

if __name__ == "__main__":
    demo.launch()
