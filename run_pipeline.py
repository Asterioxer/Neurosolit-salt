"""
One-command pipeline: filter data -> SFT -> merge -> eval gates -> (optional) private HF push.

    python run_pipeline.py --data data/dpo_pairs_mpingale_v2.jsonl --hf-repo YOUR_USERNAME/nd-solitude-sft
    python run_pipeline.py --data data/dpo_pairs_mpingale_v2.jsonl --data data/new_reviewed.jsonl --hf-repo YOUR_USERNAME/nd-solitude-sft --push

It STOPS (and never pushes) if:
  - too few clean rows survive filtering,
  - the trained model fails any eval gate (see GATES below).
Pushes are always to a PRIVATE repo.
"""

import argparse
import json
import os
import re
import subprocess
import sys

BOILERPLATE = "Staying still or doing nothing right now is also completely valid."
FLAG = re.compile(
    r"\b(you should|you need to|you must|try to|start by|make sure|look at|consider|perhaps|"
    r"you could|you can try|i wonder|it might help|it could help|i'd suggest|i would suggest)\b",
    re.IGNORECASE,
)
MOJIBAKE = re.compile("\u00c2|\u00e2\u20ac|\u00c3")
OUT = "pipeline_out"


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!])\s+", t) if len(s.strip()) > 15]


def norm_words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def ngrams(words, n):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlap(chosen, rejected):
    # share of chosen's 5-word sequences that also appear in rejected (punctuation-insensitive)
    c = ngrams(norm_words(chosen), 5)
    return len(c & ngrams(norm_words(rejected), 5)) / len(c) if c else 0.0


def filter_rows(paths, min_words):
    seen, kept, stats = set(), [], {"copied_from_rejected": 0, "mojibake": 0, "flagged": 0, "short": 0, "dup": 0}
    total = 0
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            ch = r["chosen"].replace(BOILERPLATE, "").strip()
            if overlap(ch, r["rejected"]) >= 0.4:
                stats["copied_from_rejected"] += 1
            elif any(MOJIBAKE.search(r[k]) for k in ("prompt", "chosen", "rejected")):
                stats["mojibake"] += 1
            elif "?" in ch or FLAG.search(ch):
                stats["flagged"] += 1
            elif len(ch.split()) < min_words:
                stats["short"] += 1
            elif r["prompt"].strip().lower() in seen:
                stats["dup"] += 1
            else:
                seen.add(r["prompt"].strip().lower())
                kept.append({**r, "chosen": ch})

    # Template caps: the word "still" (a known over-used template) and any 4-word phrase
    # found in more than ~4% of rows keep only their first few rows.
    still_cap = int(0.03 * len(kept))
    phrase_cap = max(3, int(0.04 * len(kept)))
    df = {}
    for r in kept:
        for g in ngrams(norm_words(r["chosen"]), 4):
            df[g] = df.get(g, 0) + 1
    over = {g for g, c in df.items() if c > phrase_cap}
    used, still_n, final = {}, 0, []
    for r in kept:
        words = norm_words(r["chosen"])
        grams = ngrams(words, 4) & over
        if "still" in words and still_n >= still_cap:
            stats["over_used_still"] = stats.get("over_used_still", 0) + 1
            continue
        if any(used.get(g, 0) >= phrase_cap for g in grams):
            stats["over_used_phrase"] = stats.get("over_used_phrase", 0) + 1
            continue
        still_n += "still" in words
        for g in grams:
            used[g] = used.get(g, 0) + 1
        final.append(r)
    return total, final, stats


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--min-rows", type=int, default=150)
    ap.add_argument("--min-words", type=int, default=12)
    ap.add_argument("--hf-repo")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="train/eval/push the LoRA adapter only (for 7B)")
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    train_file = f"{OUT}/train.jsonl"
    merged = f"{OUT}/merged"
    out_dir = f"{OUT}/adapter" if args.no_merge else merged

    total, kept, stats = filter_rows(args.data, args.min_words)
    print(f"rows in: {total}   usable: {len(kept)}   dropped: {stats}")
    if len(kept) < args.min_rows:
        sys.exit(f"STOP: only {len(kept)} usable rows (need {args.min_rows}). Add reviewed hand-written data with another --data file.")
    with open(train_file, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not args.skip_train:
        cmd = [sys.executable, "training/train_sft.py", "--base-model", args.base_model, "--dataset-file", train_file,
               "--epochs", str(args.epochs), "--max-length", str(args.max_length), "--gradient-checkpointing",
               "--adapter-dir", f"{OUT}/adapter", "--merged-dir", merged, "--ckpt-dir", f"{OUT}/ckpt"]
        if args.no_merge:
            cmd.append("--no-merge")
        run(cmd)

    who = (["--base", args.base_model, "--adapter", f"{OUT}/adapter", "--load-4bit"] if args.no_merge
           else ["--model", merged])
    run([sys.executable, "eval_models.py", *who, "--temperature", "0.4", "--samples", "4",
         "--show", "1", "--json-out", f"{OUT}/eval.json"])
    m = json.load(open(f"{OUT}/eval.json"))

    # GATES -- all must pass
    gates = {
        "no '?' in replies": m["question_marks"] == 0,
        "no soft-imperatives (<=1 allowed)": m["soft_imperative"] <= 1,
        "no lists": m["lists"] == 0,
        "no tiny replies": m["tiny"] == 0,
        "mean length 20-80 words": 20 <= m["mean_words"] <= 80,
        "no repeated closing sentence (<=15%)": m["top_closing_share"] <= 0.15,
        "no dominant opener (<=40%)": m["top_opening_share"] <= 0.40,
        "no repeated phrase across replies (<=35%)": m["top_trigram_share"] <= 0.35,
    }
    print("\nGATES")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not all(gates.values()):
        sys.exit("STOP: model failed gates. Not pushing. Fix the data (see failed gates) and re-run.")

    print("\nAll gates passed. Read the sample replies above yourself -- gates cannot judge meaning.")
    if not args.push:
        print("Not pushing (no --push). Model files are in " + out_dir)
        return
    if not args.hf_repo:
        sys.exit("--push needs --hf-repo")

    with open(f"{out_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(
            "# Experimental low-demand declarative-language model\n\n"
            "Research prototype. NOT a crisis tool and NOT for people at risk of self-harm. "
            "Route any crisis or self-harm message to a human-written response with local crisis "
            "resources BEFORE it reaches this model.\n"
        )
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.hf_repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=args.hf_repo)
    print(f"Pushed (PRIVATE) to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
