"""
Find forum-derived rows: `chosen` made of sentences copied from `rejected`,
or text with broken UTF-8 (mojibake). Reports counts and writes the rows that
look hand-written to a new file.

    python find_sliced_rows.py data/pairs_v3_clean.jsonl data/pairs_v4_handwritten.jsonl
"""

import json
import re
import sys

MOJIBAKE = re.compile("\u00c2|\u00e2\u20ac|\u00c3")  # Â, â€, Ã


def sentences(t: str):
    return [s.strip() for s in re.split(r"(?<=[.!])\s+", t) if len(s.strip()) > 15]


def overlap(chosen: str, rejected: str) -> float:
    s = sentences(chosen)
    return sum(x in rejected for x in s) / len(s) if s else 0.0


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    sliced, moji, good = [], [], []
    for r in rows:
        if overlap(r["chosen"], r["rejected"]) >= 0.4:
            sliced.append(r)
        elif any(MOJIBAKE.search(r[k]) for k in ("prompt", "chosen", "rejected")):
            moji.append(r)
        else:
            good.append(r)

    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"total rows:                       {len(rows)}")
    print(f"chosen copied from rejected:      {len(sliced)}")
    print(f"mojibake (and not sliced):        {len(moji)}")
    print(f"remaining (look hand-written):    {len(good)} -> {dst}")
    print("\nSample of remaining chosen openers:")
    for r in good[:6]:
        print("  -", r["chosen"][:110].replace("\n", " "))


if __name__ == "__main__":
    main()
