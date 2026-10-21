"""
Chat with the fine-tuned adapter, with the crisis guard in front of it.

    python chat_with_guard.py --adapter Scadiva/nd-solitude-sft-7b --region IN
    python chat_with_guard.py --adapter ./pipeline_out/adapter --region IN

Needs: huggingface login (the repo is private), a CUDA GPU, and
pip install transformers peft bitsandbytes accelerate
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from crisis_guard import guard

SYSTEM_PROMPT = (
    "Act as a low-demand, neurodiversity-affirming clinical companion grounded in "
    "Linda K. Murphy's BERMED Declarative Language, Richard Erskine's Schizoid-Process "
    "Attunement to Silence, and Johan Pahnke's NeuroACT. Never ask questions, never give "
    "imperative commands, and never pressure the user to socialize."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--region", default="IN", choices=["IN", "US", "UK", "OTHER"])
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate(text: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=150, do_sample=True,
                             temperature=args.temperature, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def on_event(e: dict) -> None:
        # level + pattern names only; never log the message text
        print(f"[guard] level={e['level']} patterns={e['patterns']}")

    print("Type a message (Ctrl+C to quit).")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue
        r = guard(text, generate, region=args.region, on_event=on_event)
        print(f"[{r['source']}] {r['reply']}")


if __name__ == "__main__":
    main()
