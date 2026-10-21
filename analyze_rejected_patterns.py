"""
Analyze the 'rejected' side of dpo_pairs_mpingale.jsonl to see what kind of
bad response the model was actually trained to avoid.

The working hypothesis: if 'rejected' examples are mostly harsh/dismissive/
clinical-cold, the model never learned that SOFT-SOUNDING advice-giving
("it might help to try...", "consider...", "perhaps...") is ALSO wrong --
which would explain why epoch-3 defaults to exactly that register on novel
prompts. This script checks that hypothesis against the real data instead
of guessing.

Usage:
    python analyze_rejected_patterns.py
    (expects dpo_pairs_mpingale.jsonl in the same directory, or pass a path)
"""

import json
import re
import sys
from collections import Counter

DEFAULT_PATH = "data/dpo_pairs_mpingale.jsonl"

QUESTION_MARK = "?"

# Soft-imperative markers: advice-giving dressed in gentle language.
# This is the register the fine-tuned model is currently defaulting to.
SOFT_IMPERATIVE_MARKERS = [
    "it might help", "it might be helpful", "it could help",
    "consider", "perhaps", "you could", "you can try",
    "it's important to", "it is important to",
    "remember to", "make sure", "try to", "try taking",
    "i'd suggest", "i would suggest", "my suggestion",
]

# Harsh/dismissive/clinical-cold markers: the OTHER kind of bad response.
HARSH_COLD_MARKERS = [
    "you should", "you need to", "you must", "have you tried",
    "just", "simply", "it's not that hard", "snap out",
    "get over it", "stop", "quit",
]

# Numbered or bulleted list detection (structural, not lexical)
LIST_PATTERN = re.compile(r"(?:^|\n)\s*(?:\d+[\.\)]|[-*•])\s+", re.MULTILINE)


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping malformed line {line_num}: {e}")
                continue
            pairs.append(obj)
    return pairs


def extract_rejected_text(obj):
    """
    Handle a few common DPO jsonl shapes:
      {"prompt": ..., "chosen": ..., "rejected": ...}
      {"prompt": ..., "chosen": [...], "rejected": [...]}  (chat-format turns)
    """
    rejected = obj.get("rejected")
    if rejected is None:
        return None
    if isinstance(rejected, str):
        return rejected
    if isinstance(rejected, list):
        # Chat-format: list of {"role":..., "content":...} -- take assistant turns
        parts = [
            turn.get("content", "")
            for turn in rejected
            if isinstance(turn, dict) and turn.get("role") == "assistant"
        ]
        if parts:
            return "\n".join(parts)
        # Fallback: just join whatever's there
        return json.dumps(rejected)
    return str(rejected)


def classify(text):
    text_lower = text.lower()
    has_question = QUESTION_MARK in text
    soft_hits = [m for m in SOFT_IMPERATIVE_MARKERS if m in text_lower]
    harsh_hits = [m for m in HARSH_COLD_MARKERS if m in text_lower]
    has_list = bool(LIST_PATTERN.search(text))
    return {
        "has_question": has_question,
        "soft_markers": soft_hits,
        "harsh_markers": harsh_hits,
        "has_list": has_list,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"Loading {path}...")
    try:
        pairs = load_pairs(path)
    except FileNotFoundError:
        print(f"  [error] file not found: {path}")
        print(f"  Usage: python analyze_rejected_patterns.py <path_to_jsonl>")
        sys.exit(1)

    print(f"Loaded {len(pairs)} pairs.\n")

    category_counts = Counter()
    soft_marker_counts = Counter()
    harsh_marker_counts = Counter()

    soft_examples = []
    harsh_examples = []
    neither_examples = []
    both_examples = []

    skipped = 0

    for obj in pairs:
        rejected_text = extract_rejected_text(obj)
        if not rejected_text:
            skipped += 1
            continue

        result = classify(rejected_text)
        is_soft = bool(result["soft_markers"])
        is_harsh = bool(result["harsh_markers"])

        for m in result["soft_markers"]:
            soft_marker_counts[m] += 1
        for m in result["harsh_markers"]:
            harsh_marker_counts[m] += 1

        if is_soft and is_harsh:
            category_counts["both"] += 1
            if len(both_examples) < 3:
                both_examples.append(rejected_text)
        elif is_soft:
            category_counts["soft_only"] += 1
            if len(soft_examples) < 3:
                soft_examples.append(rejected_text)
        elif is_harsh:
            category_counts["harsh_only"] += 1
            if len(harsh_examples) < 3:
                harsh_examples.append(rejected_text)
        else:
            category_counts["neither"] += 1
            if len(neither_examples) < 3:
                neither_examples.append(rejected_text)

        if result["has_question"]:
            category_counts["has_question_mark"] += 1
        if result["has_list"]:
            category_counts["has_list_structure"] += 1

    total_classified = len(pairs) - skipped

    print("=" * 70)
    print("BREAKDOWN of 'rejected' response patterns")
    print("=" * 70)
    print(f"Total pairs analyzed: {total_classified} (skipped {skipped} unparseable)\n")

    print(f"  Soft-imperative only (gentle advice-giving): "
          f"{category_counts['soft_only']} "
          f"({100*category_counts['soft_only']/total_classified:.1f}%)")
    print(f"  Harsh/cold/directive only:                   "
          f"{category_counts['harsh_only']} "
          f"({100*category_counts['harsh_only']/total_classified:.1f}%)")
    print(f"  Both soft and harsh markers present:          "
          f"{category_counts['both']} "
          f"({100*category_counts['both']/total_classified:.1f}%)")
    print(f"  Neither (other failure mode entirely):        "
          f"{category_counts['neither']} "
          f"({100*category_counts['neither']/total_classified:.1f}%)")
    print()
    print(f"  Contains '?':        {category_counts['has_question_mark']} "
          f"({100*category_counts['has_question_mark']/total_classified:.1f}%)")
    print(f"  Numbered/bulleted:   {category_counts['has_list_structure']} "
          f"({100*category_counts['has_list_structure']/total_classified:.1f}%)")

    print("\n" + "=" * 70)
    print("TOP soft-imperative markers found in rejected examples")
    print("=" * 70)
    for marker, count in soft_marker_counts.most_common(10):
        print(f"  {count:4d}  '{marker}'")

    print("\n" + "=" * 70)
    print("TOP harsh/cold markers found in rejected examples")
    print("=" * 70)
    for marker, count in harsh_marker_counts.most_common(10):
        print(f"  {count:4d}  '{marker}'")

    def print_samples(label, examples):
        print("\n" + "=" * 70)
        print(f"SAMPLE: {label}")
        print("=" * 70)
        if not examples:
            print("  (none found)")
        for i, ex in enumerate(examples, 1):
            snippet = ex[:300] + ("..." if len(ex) > 300 else "")
            print(f"  [{i}] {snippet}\n")

    print_samples("soft-imperative-only rejected examples", soft_examples)
    print_samples("harsh/cold-only rejected examples", harsh_examples)
    print_samples("neither pattern (other failure mode)", neither_examples)

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    soft_pct = 100 * category_counts["soft_only"] / total_classified
    harsh_pct = 100 * category_counts["harsh_only"] / total_classified
    if soft_pct < 15 and harsh_pct > 40:
        print("  Hypothesis CONFIRMED: 'rejected' examples are overwhelmingly")
        print("  harsh/cold, with little soft-imperative representation.")
        print("  --> The model never learned soft advice-giving is also wrong.")
        print("  --> Priority for new data: soft-imperative REJECTED examples,")
        print("      paired with genuinely declarative CHOSEN responses, across")
        print("      new situations (not paraphrases of existing 561 pairs).")
    elif soft_pct >= 15:
        print("  Soft-imperative rejected examples ARE present in meaningful")
        print("  numbers. The generalization failure may be more about")
        print("  situation diversity than register diversity -- the model may")
        print("  have learned the soft-vs-declarative distinction on the")
        print("  training distribution but isn't transferring it to novel")
        print("  situations. Priority: more situation variety using the SAME")
        print("  soft-imperative-rejected / declarative-chosen pairing already")
        print("  present, rather than inventing a new rejected category.")
    else:
        print("  Mixed signal -- read the printed samples above manually before")
        print("  deciding where to prioritize new data.")


if __name__ == "__main__":
    main()
