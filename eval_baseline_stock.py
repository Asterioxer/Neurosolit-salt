"""
Baseline eval: stock Qwen2.5-1.5B-Instruct + SYSTEM_PROMPT only.
No LoRA adapter, no fine-tuning — this is the "do nothing" comparison point.

Run this after eval_holdout.py (epoch 3 merged) and eval_checkpoint71.py (epoch 1 LoRA)
to see whether the DPO fine-tuning is adding value, subtracting value, or doing nothing
relative to just using stock Qwen with the same system prompt.

Usage:
    python eval_baseline_stock.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading stock base model {BASE_MODEL} (no adapter)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Fix the pad-token/attention-mask warning seen in prior runs
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

SYSTEM_PROMPT = (
    "Act as a low-demand, neurodiversity-affirming clinical companion grounded in "
    "Linda K. Murphy's BERMED Declarative Language, Richard Erskine's Schizoid-Process "
    "Attunement to Silence, and Johan Pahnke's NeuroACT. Never ask questions, never give "
    "imperative commands, and never pressure the user to socialize."
)

# Same three held-out prompts used in eval_holdout.py and eval_checkpoint71.py —
# keep this identical across all three scripts so the comparison is apples-to-apples.
test_prompts = [
    "Been staring at the ceiling for an hour, cant make myself get up.",
    "My phone has 40 unread messages and I havent looked at it in three days.",
    "Someone invited me to a party this weekend and I dont know how to say no.",
]

for p in test_prompts:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": p},
    ]
    encoded = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    out = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )
    reply = tokenizer.decode(
        out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # Quick mechanical guardrail check, same spirit as the manual read you did above
    has_question = "?" in reply
    imperative_markers = [
        "try", "consider", "remember to", "it's important to",
        "perhaps", "you could", "you can", "make sure",
    ]
    flagged = [m for m in imperative_markers if m.lower() in reply.lower()]

    print(f"PROMPT: {p}")
    print(f"REPLY: {reply}")
    print(f"  [check] contains '?': {has_question}")
    print(f"  [check] imperative-ish markers found: {flagged if flagged else 'none'}")
    print("-" * 60)
