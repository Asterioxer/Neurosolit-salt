"""
Offline tests for the pieces that do not need a model or network.

    python -m pytest tests -q        (or: python tests/test_safety.py)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arch2_declarative_ally"))
sys.path.insert(0, str(ROOT / "dataset"))

from safety import CRISIS_MESSAGE, WATCH_NOTE, assess_risk  # noqa: E402

CRISIS_CASES = [
    "I want to kill myself",
    "thinking about suicide again",
    "i dont want to be alive anymore",
    "I wish I was dead",
    "I've been cutting myself",
    "planning to end my life",
    "I don\u2019t want to live",
    "might overdose tonight",
    "they would be better off without me",
    "k i l l  m y s e l f",
    "k.i.l.l myself",
    "I don\u200bt want to live",
]
WATCH_CASES = [
    "everything feels hopeless",
    "I can't do this anymore",
    "want to disappear forever",
]
NONE_CASES = [
    "Too much personal history was shared yesterday. The urge to delete this account is strong.",
    "words gone. too loud earlier. everyone left. cant move.",
    "Another month inside my room. Society feels impossible.",
    "You must finish this report by 5 PM.",
    "",
]


def test_crisis_detected():
    for t in CRISIS_CASES:
        assert assess_risk(t) == "crisis", t


def test_watch_detected():
    for t in WATCH_CASES:
        assert assess_risk(t) == "watch", t


def test_ordinary_text_not_flagged():
    for t in NONE_CASES:
        assert assess_risk(t) == "none", t


def test_fixed_messages_follow_the_no_question_rule():
    assert "?" not in CRISIS_MESSAGE and "?" not in WATCH_NOTE
    for needle in ("14416", "988", "116 123", "findahelpline.com"):
        assert needle in CRISIS_MESSAGE


def test_dataset_filter_rejects_questions_and_imperatives():
    from build_dpo_dataset import chosen_passes_filter, heuristic_declarative_rewrite

    assert not chosen_passes_filter("Short.")
    assert not chosen_passes_filter("Why does this happen? " + "x" * 80)
    assert not chosen_passes_filter("You should rest now. " + "x" * 80)
    out = heuristic_declarative_rewrite("You should go outside! What happened?")
    assert "?" not in out and "!" not in out


def test_circuit_breaker_and_declarative_lock():
    from unittest.mock import MagicMock
    for mod in ("gradio", "transformers", "sentence_transformers"):
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()
    sys.path.insert(0, str(ROOT / "arch3_neuroact_solitude"))
    from app import check_circuit_breaker, enforce_declarative_lock

    # Common greetings must remain NOMINAL
    assert check_circuit_breaker("hello") == "NOMINAL"
    assert check_circuit_breaker("hi") == "NOMINAL"
    assert check_circuit_breaker("thanks") == "NOMINAL"

    # Engulfment panic triggers breaker
    assert check_circuit_breaker("I said too much yesterday") == "ENGULFMENT_CIRCUIT_BREAKER"
    assert check_circuit_breaker("feel exposed after that conversation") == "ENGULFMENT_CIRCUIT_BREAKER"

    # Minimal bandwidth triggers shutdown override
    assert check_circuit_breaker("cant") == "SHUTDOWN_MINIMAL_BANDWIDTH"

    # Declarative lock strictly strips interrogatives and falls back safely
    lock_out = enforce_declarative_lock("Are you feeling overwhelmed? What is your status?")
    assert "?" not in lock_out
    assert "neutral and unhurried" in lock_out or "quiet and unhurried" in lock_out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
