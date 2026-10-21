"""
Build a DPO preference dataset pairing standard counseling responses (rejected)
against low-demand, declarative responses (chosen).

One command, no hand-holding:

    python dataset/build_dpo_dataset.py --hf-username YOU            # build + save locally
    python dataset/build_dpo_dataset.py --hf-username YOU --push     # also push to the Hub

Output: data/dpo_pairs.jsonl (and optionally <username>/nd-solitude-dpo-pairs on the Hub).

Alternate sources (while waiting on gated-dataset access, or to broaden
training data): pass --source-repo and --save-local to pull from a different
corpus into its own file, then merge the JSONLs before training.

    python dataset/build_dpo_dataset.py --hf-username YOU \
        --source-repo mpingale/mental-health-chat-dataset \
        --save-local data/dpo_pairs_mpingale.jsonl

Known-good alternates and their field names (already auto-detected below):
  Amod/mental_health_counseling_conversations   Context / Response   (gated, default)
  mpingale/mental-health-chat-dataset           questionText / answerText   (ungated)

vibhorag101/phr_mental_therapy_dataset is NOT currently supported -- its rows
are a single pre-formatted Llama-2 instruction string, not separate
prompt/response fields. It would need a small parser added to build_records()
before it can be used here.

What the automation does:
  * pulls up to --max-rows counseling rows from an open corpus,
  * DROPS every row whose prompt or response contains crisis / self-harm /
    hopelessness language (same patterns as the Spaces' safety gate). The model
    must never be trained to answer that kind of text with low-demand
    validation; the Spaces handle it with a fixed, non-model response instead,
  * filters the response sentence-by-sentence, keeping only clean sentences
    (no '?', no imperative/referral/third-party-diagnosis phrasing) rather
    than rewriting flagged ones in place, then length-bounds and dedupes,
  * prints a random sample so you can spot-check in 30 seconds.

What it cannot do: the filter is heuristic, not clinical judgment. It drops
flagged sentences wholesale rather than attempting to rewrite them (an
earlier version spliced replacement clauses into sentences mid-stream,
which produced grammatically broken output -- dropping is safer). It cannot
verify a kept sentence is actually low-demand/declarative in spirit, only
that it avoids a known, necessarily incomplete list of flagged phrasings --
a fixed phrase list cannot catch every way English expresses an imperative
or a referral-out. train_dpo.py's span_penalty_score() re-checks for
residual '?' and imperative openers at training time and warns on flagged
rows; treat that count, not another spot-check round, as the real signal for
whether to keep iterating on the filter. Read at least one spot-check sample
before training on any new source. The hand-written seed pairs below were
synthesized from published literature and were NOT written or reviewed by a
clinician. Treat the output as a prototype training set.

License note: the default corpus `Amod/mental_health_counseling_conversations`
is RAIL-D licensed -- non-commercial research use is permitted; verbatim pairs
must not be altered on redistribution (this script keeps the original response
unmodified as the REJECTED side); commercial deployment requires a $100
donation to the CCC Foundation helpline. Review the license yourself.
mpingale/mental-health-chat-dataset license: verify on its dataset card before
training on it or redistributing pairs built from it.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "arch2_declarative_ally"))
from safety import assess_risk  # noqa: E402  (shared crisis patterns)

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

_FLAG_PATTERNS = ("?", "should", "need to", "must", "try to", "start by")

# Sentences matching any of these are DROPPED wholesale from the rewrite --
# never spliced or rewritten in place. Covers imperative commands, referral-
# out language, and third-party character judgment/diagnosis, all of which
# showed up in spot-checks of real source data. This list is necessarily
# incomplete -- see module docstring.
_IMPERATIVE_RE = re.compile(
    r"\b(you should|you need to|you must|try to|start by|make sure|"
    r"be prepared|be ready|"
    r"please (?:do|try|consider|reach|talk)|"
    r"i would (?:recommend|suggest)|google|provider|hotline|"
    r"talk (?:with|to) an?\b|"
    r"seems? (?:like|to be)|sounds? like|"
    r"he (?:is|seems|sounds)|she (?:is|seems|sounds)|they (?:are|seem|sound))\b",
    re.IGNORECASE,
)

_TAGLINE = " Staying still or doing nothing right now is also completely valid."
_MAX_CHOSEN_LEN = 700
_BODY_BUDGET = _MAX_CHOSEN_LEN - len(_TAGLINE)


def heuristic_declarative_rewrite(text: str) -> str:
    """
    Crude heuristic filter; output is auto-filtered afterwards too, not
    clinically validated. Keeps only WHOLE sentences that are already clean
    (no '?', no imperative/referral/third-party-judgment phrasing from
    _IMPERATIVE_RE) -- flagged sentences are DROPPED, never rewritten in
    place. (An earlier version spliced replacement clauses into sentences
    mid-stream, producing grammatically broken output; dropping is safer
    than grafting.) Sentences are joined up to a character budget and never
    cut mid-sentence; if even the first clean sentence alone would blow the
    budget, it is dropped rather than hard-truncated.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    clean = [
        s.strip() for s in sentences
        if s.strip() and not s.strip().endswith("?") and not _IMPERATIVE_RE.search(s)
    ]

    kept: list[str] = []
    running_len = 0
    for s in clean[:3]:
        candidate_len = running_len + (1 if kept else 0) + len(s)
        if candidate_len > _BODY_BUDGET:
            break
        kept.append(s)
        running_len = candidate_len

    cleaned = " ".join(kept) if kept else "This space is quiet and unhurried."
    cleaned = cleaned.replace("!", ".")
    return cleaned + _TAGLINE


def chosen_passes_filter(chosen: str) -> bool:
    if "?" in chosen or _IMPERATIVE_RE.search(chosen):
        return False
    return 60 <= len(chosen) <= _MAX_CHOSEN_LEN


def build_records(source_repo: str, source_split: str, max_rows: int):
    records = [dict(r) for r in SEED_CLINICAL_PAIRS]
    stats = {"seed": len(records), "source_rows": 0, "dropped_crisis": 0, "dropped_filter": 0, "dropped_dupe": 0, "kept": 0}

    try:
        from datasets import load_dataset

        raw_ds = load_dataset(source_repo, split=source_split)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load {source_repo} ({exc}); continuing with seed pairs only.")
        return records, stats

    seen_prompts = {r["prompt"] for r in records}
    for row in list(raw_ds)[: max(0, max_rows)]:
        context = (
            row.get("Context")
            or row.get("input")
            or row.get("context")
            or row.get("questionText")
            or row.get("Question")
            or ""
        ).strip()
        response = (
            row.get("Response")
            or row.get("output")
            or row.get("response")
            or row.get("answerText")
            or row.get("Answer")
            or ""
        ).strip()
        if not context or not response:
            continue
        stats["source_rows"] += 1

        # Crisis-adjacent rows never enter the training set.
        if assess_risk(context) != "none" or assess_risk(response) != "none":
            stats["dropped_crisis"] += 1
            continue
        if not any(flag in response.lower() for flag in _FLAG_PATTERNS):
            continue
        if context in seen_prompts:
            stats["dropped_dupe"] += 1
            continue

        chosen = heuristic_declarative_rewrite(response)
        if not chosen_passes_filter(chosen):
            stats["dropped_filter"] += 1
            continue

        seen_prompts.add(context)
        records.append({"prompt": context, "rejected": response, "chosen": chosen})
        stats["kept"] += 1

    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-username", required=True, help="Your Hugging Face username")
    parser.add_argument("--source-repo", default="Amod/mental_health_counseling_conversations")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--max-rows", type=int, default=3000, help="Max rows to pull from the source corpus")
    parser.add_argument("--output-repo", default=None, help="Defaults to <username>/nd-solitude-dpo-pairs")
    parser.add_argument("--save-local", default=str(REPO_ROOT / "data" / "dpo_pairs.jsonl"))
    parser.add_argument("--push", action="store_true", help="Push the built dataset to the HF Hub")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_repo = args.output_repo or f"{args.hf_username}/nd-solitude-dpo-pairs"
    records, stats = build_records(args.source_repo, args.source_split, args.max_rows)

    out_path = Path(args.save_local)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Built {len(records)} preference pairs -> {out_path}")
    print(f"Stats: {stats}")
    if len(records) < 100:
        print(
            "WARNING: fewer than 100 pairs. DPO on this little data mostly memorizes it. "
            "Check your internet connection / --source-repo, or expect a weak model."
        )

    rng = random.Random(args.seed)
    sample = rng.sample(records, k=min(3, len(records)))
    print("\nSpot-check sample (read these):")
    for r in sample:
        print("-" * 60)
        print("PROMPT  :", r["prompt"][:200])
        print("CHOSEN  :", r["chosen"][:300])

    if args.push:
        from datasets import Dataset

        Dataset.from_list(records).push_to_hub(output_repo)
        print(f"\nPushed to https://huggingface.co/datasets/{output_repo}")


if __name__ == "__main__":
    main()
