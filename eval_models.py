"""
Evaluate a merged model or a LoRA adapter on held-out prompts, several
samples each, and report guardrail violations + reply length.

    python eval_models.py --model ./nd_sft_merged_local
    python eval_models.py --model ./nd_dpo_merged_local_v1
    python eval_models.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter ./nd_dpo_checkpoints/checkpoint-166
    python eval_models.py --model ./nd_sft_merged_local --samples 5 --show 2
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "Act as a low-demand, neurodiversity-affirming clinical companion grounded in "
    "Linda K. Murphy's BERMED Declarative Language, Richard Erskine's Schizoid-Process "
    "Attunement to Silence, and Johan Pahnke's NeuroACT. Never ask questions, never give "
    "imperative commands, and never pressure the user to socialize."
)

PROMPTS = [
    "Been staring at the ceiling for an hour, cant make myself get up.",
    "My phone has 40 unread messages and I havent looked at it in three days.",
    "Someone invited me to a party this weekend and I dont know how to say no.",
    "The dishes have been piling up for a week and I feel awful every time I walk past them.",
    "My coworker keeps asking me to go to lunch and I keep making excuses.",
    "I have an email I need to answer and I have been avoiding it all day.",
]

MARKERS = re.compile(
    r"\b(you should|you need to|you must|try to|try taking|start by|make sure|consider|perhaps|"
    r"you could|you can try|it might help|it could help|it can help|it might be helpful|"
    r"it's important to|it is important to|remember to|i'd suggest|i would suggest|"
    r"if it helps|if it feels comfortable|if you feel ready)\b",
    re.IGNORECASE,
)
LIST_RE = re.compile(r"^\s*(\d+\.|[-*])\s", re.MULTILINE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="merged model dir")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter / checkpoint dir (used with --base)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--show", type=int, default=1, help="replies to print per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--extra-prompt", action="append", default=[], help="extra prompt to test; repeatable")
    ap.add_argument("--json-out", help="write summary stats to this JSON file")
    ap.add_argument("--load-4bit", action="store_true", help="load the base model in 4-bit (needed for 7B on a 16GB GPU)")
    args = ap.parse_args()

    if args.adapter:
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(args.base)
        kw = {}
        if args.load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="cuda", **kw)
        model = PeftModel.from_pretrained(model, args.adapter)
        name = args.adapter
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
        name = args.model
    model.eval()

    total = q = imp = lst = tiny = 0
    words = []
    replies = []
    for p in PROMPTS + args.extra_prompt:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        mask = torch.ones_like(ids)
        print(f"\nPROMPT: {p}")
        for i in range(args.samples):
            out = model.generate(
                ids, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, pad_token_id=tok.eos_token_id,
            )
            reply = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            n = len(reply.split())
            total += 1
            q += "?" in reply
            imp += bool(MARKERS.search(reply))
            lst += bool(LIST_RE.search(reply))
            tiny += n < 10
            words.append(n)
            replies.append(reply)
            if i < args.show:
                print(f"  REPLY ({n}w): {reply}")

    print("\n" + "=" * 60)
    print(f"MODEL: {name}   samples: {total}")
    print(f"  contains '?':            {q}/{total}")
    print(f"  soft-imperative marker:  {imp}/{total}")
    print(f"  numbered/bulleted list:  {lst}/{total}")
    print(f"  tiny reply (<10 words):  {tiny}/{total}")
    print(f"  mean reply length:       {sum(words) / len(words):.0f} words")
    print("Good = low on the first four AND mean length near 40-60 words.")

    if args.json_out:
        last = [re.split(r"(?<=[.!])\s+", r.strip())[-1] for r in replies if r.strip()]
        first = [" ".join(r.lower().split()[:3]) for r in replies if r.strip()]
        top = lambda xs: max(xs.count(x) for x in set(xs)) / len(xs) if xs else 0.0
        tri = []
        for r in replies:
            w = re.findall(r"[a-z0-9']+", r.lower())
            tri.append({" ".join(w[i:i + 3]) for i in range(len(w) - 2)})
        cnt = {}
        for s in tri:
            for g in s:
                cnt[g] = cnt.get(g, 0) + 1
        top_tri = max(cnt.values()) / len(tri) if cnt else 0.0
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "samples": total, "question_marks": q, "soft_imperative": imp, "lists": lst,
                "tiny": tiny, "mean_words": sum(words) / len(words),
                "top_closing_share": top(last), "top_opening_share": top(first),
                "top_trigram_share": top_tri,
            }, f, indent=2)


if __name__ == "__main__":
    main()
