"""
Validate bulk-generated round-2 DPO pairs against the spec in round2_data_spec.md
BEFORE merging into the training file. Catches malformed pairs fast so you're not
re-training on bad data.

Expects a JSONL file with the same shape as dpo_pairs_mpingale.jsonl:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
(also accepts chat-format list-of-turns for chosen/rejected, same as the
analyze_rejected_patterns.py script)

Checks enforced:
  - rejected: zero '?'
  - chosen:   zero '?'
  - rejected: contains >=1 soft-imperative marker
  - chosen:   contains ZERO imperative-toward-user markers
  - neither side has numbered/bulleted list structure
  - flags near-duplicate prompts (simple normalized-text match) across the new set

Usage:
    python validate_round2_pairs.py new_pairs_round2.jsonl
    python validate_round2_pairs.py new_pairs_round2.jsonl --merge-check dpo_pairs_mpingale.jsonl
"""

import json
import re
import sys
import argparse
from collections import Counter

SOFT_IMPERATIVE_MARKERS = [
    "it might help", "it might be helpful", "it could help", "it can help",
    "consider", "perhaps", "you could", "you can try",
    "it's important to", "it is important to",
    "remember to", "make sure", "try to", "try taking",
    "i'd suggest", "i would suggest", "my suggestion",
]

# Any of these in the CHOSEN side is a spec violation -- chosen must be pure
# declarative, no direction at the user at all.
IMPERATIVE_MARKERS_FORBIDDEN_IN_CHOSEN = SOFT_IMPERATIVE_MARKERS + [
    "you should", "you need to", "you must", "try",
    "do this", "don't forget",
]

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
            pairs.append((line_num, obj))
    return pairs


def extract_text(field_value):
    if field_value is None:
        return ""
    if isinstance(field_value, str):
        return field_value
    if isinstance(field_value, list):
        parts = [
            turn.get("content", "")
            for turn in field_value
            if isinstance(turn, dict) and turn.get("role") == "assistant"
        ]
        return "\n".join(parts) if parts else json.dumps(field_value)
    return str(field_value)


def normalize_for_dup_check(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def validate_pair(line_num, obj):
    errors = []
    warnings = []

    prompt = obj.get("prompt", "")
    chosen = extract_text(obj.get("chosen"))
    rejected = extract_text(obj.get("rejected"))

    if not prompt:
        errors.append("missing 'prompt' field")
    if not chosen:
        errors.append("missing 'chosen' field")
    if not rejected:
        errors.append("missing 'rejected' field")

    if chosen and "?" in chosen:
        errors.append("chosen contains '?' (must be zero)")
    if rejected and "?" in rejected:
        errors.append("rejected contains '?' (spec requires zero for this round)")

    if rejected:
        soft_hits = [m for m in SOFT_IMPERATIVE_MARKERS if m in rejected.lower()]
        if not soft_hits:
            errors.append("rejected has no soft-imperative marker (spec requires >=1)")

    if chosen:
        bad_hits = [m for m in IMPERATIVE_MARKERS_FORBIDDEN_IN_CHOSEN if m in chosen.lower()]
        if bad_hits:
            errors.append(f"chosen contains forbidden imperative marker(s): {bad_hits}")

    if chosen and LIST_PATTERN.search(chosen):
        warnings.append("chosen has numbered/bulleted structure (spec says avoid)")
    if rejected and LIST_PATTERN.search(rejected):
        warnings.append("rejected has numbered/bulleted structure (spec says avoid)")

    # length sanity -- chosen should be short/declarative, not a wall of text
    if chosen and len(chosen.split()) > 60:
        warnings.append(f"chosen is {len(chosen.split())} words -- spec wants 1-2 short sentences")

    return errors, warnings, prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("new_file", help="Path to new round-2 pairs JSONL")
    parser.add_argument("--merge-check", help="Path to existing dataset JSONL, to check for cross-file duplicate prompts")
    args = parser.parse_args()

    print(f"Loading {args.new_file}...")
    pairs = load_pairs(args.new_file)
    print(f"Loaded {len(pairs)} new pairs.\n")

    all_errors = []
    all_warnings = []
    prompt_norms = {}
    clean_count = 0

    for line_num, obj in pairs:
        errors, warnings, prompt = validate_pair(line_num, obj)
        if errors:
            all_errors.append((line_num, prompt, errors))
        else:
            clean_count += 1
        if warnings:
            all_warnings.append((line_num, prompt, warnings))

        norm = normalize_for_dup_check(prompt)
        if norm in prompt_norms:
            all_warnings.append((
                line_num, prompt,
                [f"near-duplicate of line {prompt_norms[norm]}"]
            ))
        else:
            prompt_norms[norm] = line_num

    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Clean pairs (no errors):  {clean_count} / {len(pairs)}")
    print(f"Pairs with errors:        {len(all_errors)}")
    print(f"Pairs with warnings only: {len(all_warnings) - len([e for e in all_errors])}")

    if all_errors:
        print("\n" + "-" * 70)
        print("ERRORS (fix or remove these before merging)")
        print("-" * 70)
        for line_num, prompt, errors in all_errors:
            print(f"\n  Line {line_num}: {prompt[:80]}")
            for e in errors:
                print(f"    [ERROR] {e}")

    if all_warnings:
        print("\n" + "-" * 70)
        print("WARNINGS (review, not necessarily blocking)")
        print("-" * 70)
        for line_num, prompt, warnings in all_warnings:
            print(f"\n  Line {line_num}: {prompt[:80]}")
            for w in warnings:
                print(f"    [WARN] {w}")

    # Cross-file duplicate check against existing dataset
    if args.merge_check:
        print("\n" + "=" * 70)
        print(f"CROSS-FILE DUPLICATE CHECK against {args.merge_check}")
        print("=" * 70)
        existing = load_pairs(args.merge_check)
        existing_norms = set()
        for _, obj in existing:
            existing_norms.add(normalize_for_dup_check(obj.get("prompt", "")))

        cross_dupes = 0
        for line_num, obj in pairs:
            norm = normalize_for_dup_check(obj.get("prompt", ""))
            if norm in existing_norms:
                cross_dupes += 1
                print(f"  [DUP] line {line_num} prompt already exists in {args.merge_check}: "
                      f"{obj.get('prompt','')[:80]}")
        if cross_dupes == 0:
            print("  No cross-file duplicates found.")
        else:
            print(f"\n  {cross_dupes} prompt(s) already exist in the training set -- remove before merging.")

    print("\n" + "=" * 70)
    if all_errors:
        print(f"RESULT: {len(all_errors)} pair(s) FAILED validation. Fix before merging.")
        sys.exit(1)
    else:
        print(f"RESULT: All {len(pairs)} pairs passed validation. Safe to merge.")
        sys.exit(0)


if __name__ == "__main__":
    main()
