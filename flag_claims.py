"""
Scan `chosen` text for interpretive claims -- statements about WHY the user's body or
mind is doing something, or what other people think/feel -- which the model
generalises into confident, sometimes false, explanations.

    python flag_claims.py data/new_reviewed_batch1.jsonl data/new_reviewed_batch2.jsonl data/new_reviewed_batch3.jsonl
    python flag_claims.py data/new_reviewed_batch2.jsonl --show body_mind
"""

import argparse
import json
import re

PATTERNS = {
    "body_mind": r"\bthe (body|mind|brain)\b|\b(body|mind|brain)'s\b",
    "reaction_claim": r"\b(is|are) (a|the) (reaction|response|way|sign|signal)\b|\bis the (body|mind)\b",
    "hedged_generalisation": r"\b(tends? to|often|usually|common(ly)?|most people|many people|everyone|anyone)\b",
    "other_people": r"\b(host|coworkers?|friends?|sender|brother|aunt|landlord|they|them|others|people)\b[^.]{0,50}\b(understand|think|feel|care|want|expect|notice|mind|see|hear)s?\b",
}
RX = {k: re.compile(v, re.IGNORECASE) for k, v in PATTERNS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", choices=list(PATTERNS), help="print the matching rows")
    args = ap.parse_args()

    for path in args.files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        print(f"\n{path}  ({len(rows)} rows)")
        for name, rx in RX.items():
            hits = [r for r in rows if rx.search(r["chosen"])]
            print(f"  {name:24s} {len(hits):3d}  ({100 * len(hits) // max(len(rows), 1)}%)")
            if args.show == name:
                for r in hits:
                    print(f"      - {r['prompt'][:60]}\n        {r['chosen'][:220]}")
        any_hit = [r for r in rows if any(rx.search(r["chosen"]) for rx in RX.values())]
        print(f"  rows with ANY flag:      {len(any_hit):3d}  ({100 * len(any_hit) // max(len(rows), 1)}%)")


if __name__ == "__main__":
    main()
