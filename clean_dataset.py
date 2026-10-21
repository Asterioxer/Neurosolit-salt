"""
Clean the DPO/SFT pair file:
  1. Strip the boilerplate closing sentence from `chosen` (keep it in ~3% of rows).
  2. Pull out chosen rows that contain '?' or imperative/suggestion phrasing
     into a separate review file instead of training on them.
  3. Drop rows whose chosen text is too short after stripping.
  4. Report the most common openers so template repetition is visible.

    python clean_dataset.py data/dpo_pairs_mpingale_v2.jsonl data/pairs_v3_clean.jsonl
"""

import argparse
import collections
import json
import random
import re

BOILERPLATE = "Staying still or doing nothing right now is also completely valid."
FLAG = re.compile(
    r"\b(you should|you need to|you must|try to|start by|make sure|look at|consider|perhaps|"
    r"you could|you can try|i wonder)\b",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-frac", type=float, default=0.03, help="fraction of rows keeping the closing sentence")
    ap.add_argument("--min-words", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()]

    kept, flagged, too_short = [], [], []
    stripped = retained = 0
    for r in rows:
        ch = r["chosen"].strip()
        if "?" in ch or FLAG.search(ch):
            flagged.append(r)
            continue
        if BOILERPLATE in ch:
            if rng.random() < args.keep_frac:
                retained += 1
            else:
                ch = ch.replace(BOILERPLATE, "").strip()
                stripped += 1
        if len(ch.split()) < args.min_words:
            too_short.append(r)
            continue
        kept.append({**r, "chosen": ch})

    with open(args.dst, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    review = args.dst.replace(".jsonl", "_REVIEW.jsonl")
    with open(review, "w", encoding="utf-8", newline="\n") as f:
        for r in flagged + too_short:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    words = sorted(len(r["chosen"].split()) for r in kept)
    print(f"input rows:            {len(rows)}")
    print(f"closing sentence cut:  {stripped}   kept in: {retained}")
    print(f"flagged (? / imperative): {len(flagged)}   too short after strip: {len(too_short)}  -> {review}")
    print(f"output rows:           {len(kept)} -> {args.dst}")
    print(f"chosen words: min {words[0]}  median {words[len(words) // 2]}  max {words[-1]}")
    print("\nMost common openers (first 4 words):")
    for k, v in collections.Counter(" ".join(r["chosen"].split()[:4]).lower() for r in kept).most_common(8):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
