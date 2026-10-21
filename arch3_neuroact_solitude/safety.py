"""
Hard crisis gate, run BEFORE any model call.

Mirror of arch1_liminal_space/crisis.js -- keep both pattern lists in sync
(scripts/check_sync.py verifies this). Each Space folder carries its own copy
of this file so it deploys standalone.

This is a keyword screen, NOT a clinical risk assessment. It will miss
indirect or novel phrasing, and it over-triggers on negated phrasing
("I am not going to kill myself") on purpose: for this use case a false alarm
is far cheaper than a miss. It needs review by a clinician before public use.
"""

import re
import unicodedata

CRISIS_PATTERNS = [
    r"\bsuicid\w*",
    r"\bkill(?:ing)? ?my ?self\b",
    r"\bkill me\b",
    r"\bend(?:ing)? (?:it all|my life|my own life|things for good)\b",
    r"\btake my (?:own )?life\b",
    r"\bwant(?:ed|s)? to die\b",
    r"\bwish i (?:was|were) (?:dead|never born)\b",
    r"\bwish i (?:could |would )?(?:just )?(?:die|not wake up|never wake up)\b",
    r"\b(?:do not|don'?t|dont) want to (?:be alive|live|exist|be here anymore)\b",
    r"\bno reason to (?:live|go on|be alive)\b",
    r"\b(?:hurt|harm|cut|cutting|burn|burning|punish) ?(?:my ?self)\b",
    r"\bself[- ]?harm\w*",
    r"\boverdos\w*",
    r"\bhang(?:ing)? ?my ?self\b",
    r"\bjump(?:ing)? (?:off|from) (?:a |the )?(?:bridge|building|roof|balcony)\b",
    r"\bbetter off dead\b",
    r"\bend my suffering\b",
    r"\bunalive\w*",
    r"\bthink(?:ing)? (?:about|of) (?:ending|killing) (?:it|myself)\b",
    r"\b(?:took|taking) (?:too many|a lot of|a bunch of) (?:pills|tablets|meds)\b",
    r"\bbetter off without me\b",
    r"\b(?:such )?a burden (?:to|on) (?:everyone|everybody|my family|them|others|all of you)\b",
    r"\b(?:don'?t|dont|do not) want to wake up\b",
    r"\b(?:don'?t|dont|can'?t|cant) see (?:the|any) (?:point|reason) (?:in|to|of) (?:living|life|going on|go on)\b",
    r"\bmarna chah(?:ta|ti) h(?:u|oon|un)\b",
    r"\bjeene ka (?:mann|man) nahi\b",
    r"\baatm ?hatya\b",
    r"आत्महत्या",
    r"मरना चाहत",
]

WATCH_PATTERNS = [
    r"\bno point (?:in|to) (?:living|going on|anything)\b",
    r"\bcan'?t (?:do|take) this anymore\b",
    r"\bcannot (?:do|take) this anymore\b",
    r"\bgive up on (?:life|everything)\b",
    r"\bhopeless\b",
    r"\bwant to disappear forever\b",
    r"\bdisappear forever\b",
    r"\bnot wake up\b",
    r"\b(?:don'?t|dont|can'?t|cant) see (?:the|any) (?:point|reason) (?:in|to|of) anything\b",
]

_CRISIS_RES = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]
_WATCH_RES = [re.compile(p, re.IGNORECASE) for p in WATCH_PATTERNS]

CRISIS_MESSAGE = (
    "What is written here sounds like it may involve thoughts of ending life or hurting oneself. "
    "This tool is not built for that, so it is stepping aside instead of responding.\n\n"
    "Support from a trained person is available at any hour:\n"
    "- India: Tele-MANAS 14416 or 1-800-891-4416 (free, 24x7, many languages)\n"
    "- US: call or text 988\n"
    "- UK and Ireland: Samaritans 116 123\n"
    "- Elsewhere: findahelpline.com lists lines by country\n\n"
    "If there is immediate danger, the local emergency number reaches help fastest. "
    "Contacting one trusted person who can be physically nearby is also an option. "
    "No reply to this message is needed."
)

WATCH_NOTE = (
    "A note, offered once, requiring nothing: if thoughts of not wanting to be alive are present, "
    "trained support is available at any hour. India: Tele-MANAS 14416. US: 988. UK and Ireland: "
    "Samaritans 116 123. Elsewhere: findahelpline.com."
)


_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_QUOTES = re.compile(r"['\u2019\u2018`]")
_SPACED = re.compile(r"(?:\b\w\s+){3,}\w\b")
_PUNCT_COLLAPSE = re.compile(r"(?<=\b\w)[.\-_/](?=\w\b)")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _ZERO_WIDTH.sub("", t)
    t = _PUNCT_COLLAPSE.sub("", t)
    t = _QUOTES.sub("'", t)
    t = _SPACED.sub(lambda m: re.sub(r"\s+", "", m.group(0)), t)
    return t


def assess_risk(text: str) -> str:
    """Returns "crisis", "watch" or "none"."""
    t = normalize(text)
    if any(r.search(t) for r in _CRISIS_RES):
        return "crisis"
    if any(r.search(t) for r in _WATCH_RES):
        return "watch"
    return "none"
