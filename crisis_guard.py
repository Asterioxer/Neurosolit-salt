"""
crisis_guard.py -- runs BEFORE the model. If a message looks like suicidal
thoughts or self-harm, the model is never called; a fixed, human-written
response with real help lines is returned instead.

Why this exists: the model is trained to write calm declarative sentences and
never ask questions. That style is wrong for someone in danger, and a small
fine-tuned model cannot be trusted to recognise a crisis reliably.

Levels
  high    explicit intent / ideation / self-harm        -> fixed crisis response
  medium  passive ideation ("better off without me")    -> softer fixed response
  watch   general despair ("I can't do this anymore")   -> NOT routed; returned so
                                                           you can log/count it
  None    nothing found                                 -> model replies normally

Design choices
  * Recall over precision: a false alarm costs one extra message with help
    lines; a miss can cost far more. Tune `route_medium` if medium is too noisy.
  * Never log the message text. Log only level + pattern names (on_event).
  * The crisis wording below is a DRAFT. Have a clinician or crisis-line
    professional review it, and have native speakers review the Hindi/Hinglish
    patterns (and add Odia and other languages your users write in).
  * Re-verify every phone number before release; helplines change.

Usage
    from crisis_guard import guard
    result = guard(user_text, generate=my_model_fn, region="IN")
    send(result["reply"])          # result["source"] is "crisis_guard" or "model"
"""

import re
import unicodedata

# ---------------------------------------------------------------- help lines
REGIONS = {
    "IN": {
        "emergency": "112",
        "lines": [
            "Tele-MANAS (Government of India): call 14416 or 1-800-891-4416, free, 24/7, in many languages. "
            "Website: telemanas.mohfw.gov.in",
        ],
    },
    "US": {
        "emergency": "911",
        "lines": ["988 Suicide & Crisis Lifeline: call or text 988, free, 24/7"],
    },
    "UK": {
        "emergency": "999",
        "lines": ["Samaritans: call 116 123, free, 24/7"],
    },
    "OTHER": {
        "emergency": "your local emergency number",
        "lines": ["findahelpline.com lists crisis lines by country"],
    },
}

# ------------------------------------------------------------------ patterns
# Applied to NORMALIZED text: lowercase, apostrophes removed ("dont", "im"),
# ASCII punctuation turned into spaces, spaced-out letters re-joined.
_HIGH = {
    "suicide_word": r"suicid|\bunaliv",
    "kill_myself": r"\bkill(?:ing)? ?my ?self\b",
    "end_life": r"\b(?:end|ending|take|taking) my (?:own )?life\b",
    "end_it_all": r"\bend(?:ing)? it all\b",
    "thinking_ending_it": r"\b(?:thinking|thought|thoughts|think) (?:about|of) (?:ending|killing) (?:it|myself|my)\b",
    "want_to_die": r"\b(?:want|wants|wanna|wanted|ready|need|planning|plan) (?:to )?(?:die|be dead)\b",
    "wish_dead": r"\bwish(?:ed)? (?:i|that i) (?:was|were|am|could|would) (?:just )?(?:dead|not alive|never born|die)\b",
    "better_dead": r"\b(?:better off dead|rather be dead|rather die)\b",
    "dont_want_live": r"\b(?:dont|do not|didnt|no longer|never) (?:really )?(?:want|wanna) (?:to )?(?:live|be alive|exist|keep living)\b",
    "self_harm_active": r"\b(?:cutting|harming|hurting|burning|punishing) my ?self\b",
    "self_harm_intent": r"\b(?:want|wanna|going|need|have|urge) to (?:cut|hurt|harm|burn|punish) my ?self\b",
    "self_harm_word": r"\bself ?harm",
    "overdose": r"\boverdos|\btook too many (?:pills|tablets|meds)\b|\btaking too many (?:pills|tablets|meds)\b",
    "hinglish": (r"\bmarna chah(?:ta|ti) h(?:u|oon|un|oo)\b|\bmar jaun(?:ga|gi)?\b|\bmar jana chah"
                 r"|\bjeena nahi chah|\bjeene ka (?:mann|man|mood) nahi\b|\bjaan de dun(?:ga|gi)\b"
                 r"|\bkhud ko (?:khatam|maar)|\baatm ?hatya|\batm ?hatya"),
    "hindi_script": r"आत्महत्या|मरना चाहत|जीना नहीं चाहत|जीने का मन नहीं|खुद को खत्म|खुद को मार",
}

_MEDIUM = {
    "no_point": (r"\b(?:no|dont see (?:the|any)|cant see (?:the|any)|lost (?:the|all|my)|see no) "
                 r"(?:point|reason|will)(?: (?:in|to|of|for))? "
                 r"(?:anything|living|life|going on|go on|being here|carrying on|carry on|continuing|continue|live|keep going)\b"),
    "dont_want_here": r"\b(?:dont|do not|no longer|never) (?:really )?(?:want|wanna) (?:to )?(?:be here|wake up|go on|carry on|keep going|be around)\b",
    "better_without_me": r"\bbetter off without me\b|\bburden (?:to|on) (?:everyone|everybody|my family|them|all of you|others)\b|\b(?:am|im|such|just) a burden\b",
    "disappear": r"\b(?:want|wanna|wish) (?:to|i could) (?:disappear|vanish|not exist|stop existing|go to sleep and never wake up)\b",
    "kill_me": r"\bkill me\b",
    "wish_not_wake": r"\bwish i (?:would|could) (?:just )?(?:not wake up|never wake up)\b",
}

_WATCH = {
    "cant_anymore": r"\b(?:cant|cannot) (?:do this|take (?:it|this)|go on|keep going|carry on) anymore\b",
    "tired_of_everything": r"\b(?:tired of everything|sick of everything|giving up)\b",
    "hopeless": r"\bhopeless\b|\bno hope\b|\bworthless\b|\bwhats the point\b",
}

_HIGH = {k: re.compile(v) for k, v in _HIGH.items()}
_MEDIUM = {k: re.compile(v) for k, v in _MEDIUM.items()}
_WATCH = {k: re.compile(v) for k, v in _WATCH.items()}

# ------------------------------------------------------------ normalization
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_QUOTES = re.compile("['\u2019\u2018`]")
# ASCII punctuation only: never strip Devanagari vowel signs.
_PUNCT = re.compile(r"[!\"#$%&()*+,\-./:;<=>?@\[\\\]^_{|}~\u2013\u2014\u2026]")
_SPACED = re.compile(r"(?:\b\w\s+){3,}\w\b")  # "k i l l  m y s e l f"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = _ZERO_WIDTH.sub("", t)
    t = _QUOTES.sub("", t)
    t = _PUNCT.sub(" ", t)
    t = _SPACED.sub(lambda m: re.sub(r"\s+", "", m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


def detect(text: str) -> dict:
    """Return {"level": "high"|"medium"|"watch"|None, "matches": [pattern names]}."""
    t = normalize(text)
    for level, table in (("high", _HIGH), ("medium", _MEDIUM), ("watch", _WATCH)):
        hits = [name for name, rx in table.items() if rx.search(t)]
        if hits:
            return {"level": level, "matches": hits}
    return {"level": None, "matches": []}


# ----------------------------------------------------------------- responses
def crisis_message(level: str, region: str = "IN") -> str:
    r = REGIONS.get(region, REGIONS["OTHER"])
    lines = "\n".join(f"- {x}" for x in r["lines"])
    if level == "high":
        return (
            "I'm really glad you said this out loud. It sounds like you are in a lot of pain, and that "
            "deserves support from a real person, not only from me.\n\n"
            f"If you think you might act on these thoughts, or you are in danger right now, please call "
            f"{r['emergency']} or go to the nearest hospital. If someone is near you, please tell them or "
            "ask them to stay with you.\n\n"
            f"You can also talk to someone trained for exactly this, right now:\n{lines}\n\n"
            "I'm still here to talk as well."
        )
    return (
        "That sounds really heavy, and it matters. You don't have to be at the edge of a crisis for it "
        "to be worth talking to someone.\n\n"
        f"If these feelings ever turn into thoughts of ending your life, or you feel unsafe, please reach "
        f"out right away: call {r['emergency']}, or contact:\n{lines}\n\n"
        "I'm still here to talk as well."
    )


def guard(user_text, generate, region="IN", route_medium=True, on_event=None) -> dict:
    """
    generate: function(str) -> str that calls your model.
    on_event: optional callback({"level", "patterns"}) -- never receives the message text.
    """
    result = detect(user_text)
    level = result["level"]
    if level == "high" or (level == "medium" and route_medium):
        if on_event:
            on_event({"level": level, "patterns": result["matches"]})
        return {"reply": crisis_message(level, region), "source": "crisis_guard", "level": level}
    if level == "watch" and on_event:
        on_event({"level": level, "patterns": result["matches"]})
    return {"reply": generate(user_text), "source": "model", "level": level}
