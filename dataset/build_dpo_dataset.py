"""
Build a DPO preference dataset pairing standard counseling-dataset responses
(rejected) against low-demand, solitude-affirming declarative responses
(chosen).

Usage:
    python build_dpo_dataset.py --hf-username YOUR_USERNAME --push

License note: the optional base corpus `Amod/mental_health_counseling_conversations`
is governed by the RAIL-D license. Non-commercial research filtering/subsetting
is permitted; verbatim pairs must not be altered on redistribution; commercial
deployment requires a $100 donation to the CCC Foundation helpline. This script
only extracts context (x) and uses the original response as the REJECTED side
(unmodified), which should satisfy the "do not alter verbatim pairs" condition
during redistribution -- but review the license yourself before shipping
anything built from this.

IMPORTANT: the heuristic in `heuristic_declarative_rewrite()` is a crude
regex-based converter, not a clinical judgment. Every row it produces should
be reviewed by a human before being used for training. Do not treat the
output of this script as a finished, validated dataset.
"""

import argparse
import re

from datasets import Dataset, load_dataset

# -----------------------------------------------------------------------
# Hand-written seed pairs. These were synthesized from the published
# clinical literature on BERMED declarative language, schizoid-process
# theory, and NeuroACT -- they were NOT written or reviewed by a clinician.
# Review before use. See docs/ROADMAP.md hardening backlog.
# -----------------------------------------------------------------------
SEED_CLINICAL_PAIRS = [
    {
        "prompt": (
            "Too much personal history was shared yesterday. The urge to "
            "delete this account and disappear is overwhelming."
        ),
        "rejected": (
            "Please don't disappear! Sharing your feelings is a brave step "
            "toward healing and building trust. Why do you feel afraid of "
            "opening up?"
        ),
        "chosen": (
            "Stepping back after sharing makes complete sense. Nothing "
            "shared yesterday creates any obligation today, and the context "
            "can be cleared at any time. This space remains neutral and "
            "unhurried."
        ),
    },
    {
        "prompt": (
            "Sitting on the floor for four hours. Thirsty and need to "
            "shower, but the thought of having to do it makes my skin "
            "crawl."
        ),
        "rejected": (
            "It sounds like executive dysfunction! Let's break it down: "
            "1. Stand up right now. 2. Drink a glass of water. 3. Take a "
            "quick 5-minute shower. You can do this!"
        ),
        "chosen": (
            "Bodies sometimes just need to stay still on the floor for a "
            "while when the elastic band has been stretched tight. Water "
            "and showers don't have a deadline. Staying put is completely "
            "fine."
        ),
    },
    {
        "prompt": "words gone. too loud earlier. everyone finally left the house. cant move.",
        "rejected": (
            "I'm sorry you had a sensory overload! Try practicing deep "
            "breathing right now. What triggered the noise earlier, and "
            "how can you prevent it next time?"
        ),
        "chosen": (
            "The house is quiet now. No words or answers are needed here. "
            "The quiet will stay as long as needed."
        ),
    },
    {
        "prompt": (
            "Another month inside my room. Only awake at 3 AM when the "
            "street is dead silent. Society feels impossible."
        ),
        "rejected": (
            "Isolating yourself for a month is unhealthy and worsens "
            "depression. Try going for a morning walk tomorrow and "
            "reaching out to an old friend!"
        ),
        "chosen": (
            "Three in the morning has a very predictable, low-sensory "
            "stillness to it. When the outside world carries high "
            "friction, a quiet room protects energy. I wonder what "
            "solitary projects or interests have been holding attention "
            "tonight."
        ),
    },
]

# Patterns used to decide whether a dataset row's response is worth
# converting (i.e. it contains a question or an imperative marker).
_FLAG_PATTERNS = ("?", "should", "need to", "must", "try to", "start by")

_REPLACEMENTS = [
    (r"\bYou should\b", "One option that sometimes exists is to"),
    (r"\bYou need to\b", "It can sometimes help to"),
    (r"\bYou must\b", "It is possible to"),
    (r"\bTry to\b", "I wonder if it might feel manageable to"),
    (r"\bStart by\b", "Some people notice a shift when"),
]


def heuristic_declarative_rewrite(text: str) -> str:
    """
    Crude regex-based conversion of imperative/interrogative counselor text
    into a declarative-leaning observation. This is NOT a clinical
    rewrite -- it is a starting point for human review, used to bootstrap
    volume in the dataset. Every output should be read before training on
    it.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    non_questions = [s for s in sentences if not s.strip().endswith("?")]
    cleaned = " ".join(non_questions[:3]) if non_questions else "This space is quiet and unhurried."

    for pattern, repl in _REPLACEMENTS:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    return cleaned + " Staying still or doing nothing right now is also completely valid."


def build_dataset(source_repo: str, source_split: str, max_rows: int) -> Dataset:
    records = list(SEED_CLINICAL_PAIRS)

    try:
        raw_ds = load_dataset(source_repo, split=source_split)
    except Exception as exc:  # noqa: BLE001 - surface clearly, don't crash the whole run
        print(f"Could not load {source_repo} ({exc}); continuing with seed pairs only.")
        raw_ds = []

    for row in list(raw_ds)[: max(0, max_rows)]:
        context = row.get("Context") or row.get("input") or row.get("context") or ""
        response = row.get("Response") or row.get("output") or row.get("response") or ""
        if not context or not response:
            continue
        lowered = response.lower()
        if any(flag in lowered for flag in _FLAG_PATTERNS):
            records.append(
                {
                    "prompt": context.strip(),
                    "rejected": response.strip(),
                    "chosen": heuristic_declarative_rewrite(response),
                }
            )

    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-username", required=True, help="Your Hugging Face username")
    parser.add_argument(
        "--source-repo",
        default="Felladrin/pretrain-mental-health-counseling-conversations",
        help="Open counseling dataset to draw rejected/base rows from",
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--max-rows", type=int, default=500, help="Max rows to pull from the source corpus")
    parser.add_argument("--output-repo", default=None, help="Defaults to <username>/nd-solitude-dpo-pairs")
    parser.add_argument("--push", action="store_true", help="Push the built dataset to the HF Hub")
    parser.add_argument("--save-local", default=None, help="Optional local path to save the dataset (JSON)")
    args = parser.parse_args()

    output_repo = args.output_repo or f"{args.hf_username}/nd-solitude-dpo-pairs"

    ds = build_dataset(args.source_repo, args.source_split, args.max_rows)
    print(f"Built {len(ds)} preference pairs ({len(SEED_CLINICAL_PAIRS)} hand-written seed pairs included).")
    print("REVIEW THESE BEFORE TRAINING -- the heuristic converter is not clinically validated.")

    if args.save_local:
        ds.to_json(args.save_local)
        print(f"Saved to {args.save_local}")

    if args.push:
        ds.push_to_hub(output_repo)
        print(f"Pushed to https://huggingface.co/datasets/{output_repo}")


if __name__ == "__main__":
    main()
