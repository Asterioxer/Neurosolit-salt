// Hard crisis gate. Runs BEFORE any model call, entirely client-side.
// Mirror of shared/crisis.py -- keep the two pattern lists in sync
// (scripts/check_sync.py verifies this).
//
// This is a keyword screen, NOT a clinical risk assessment. It will miss
// indirect or novel phrasing and it over-triggers on negated phrasing
// ("I am not going to kill myself") on purpose: for this use case a false
// alarm is far cheaper than a miss. It needs review by a clinician.

export const CRISIS_PATTERNS = [
  "\\bsuicid\\w*",
  "\\bkill(?:ing)? ?my ?self\\b",
  "\\bkill me\\b",
  "\\bend(?:ing)? (?:it all|my life|my own life|things for good)\\b",
  "\\btake my (?:own )?life\\b",
  "\\bwant(?:ed|s)? to die\\b",
  "\\bwish i (?:was|were) (?:dead|never born)\\b",
  "\\bwish i (?:could |would )?(?:just )?(?:die|not wake up|never wake up)\\b",
  "\\b(?:do not|don'?t|dont) want to (?:be alive|live|exist|be here anymore)\\b",
  "\\bno reason to (?:live|go on|be alive)\\b",
  "\\b(?:hurt|harm|cut|cutting|burn|burning|punish) ?(?:my ?self)\\b",
  "\\bself[- ]?harm\\w*",
  "\\boverdos\\w*",
  "\\bhang(?:ing)? ?my ?self\\b",
  "\\bjump(?:ing)? (?:off|from) (?:a |the )?(?:bridge|building|roof|balcony)\\b",
  "\\bbetter off dead\\b",
  "\\bend my suffering\\b",
  "\\bunalive\\w*",
  "\\bthink(?:ing)? (?:about|of) (?:ending|killing) (?:it|myself)\\b",
  "\\b(?:took|taking) (?:too many|a lot of|a bunch of) (?:pills|tablets|meds)\\b",
  "\\bbetter off without me\\b",
  "\\b(?:such )?a burden (?:to|on) (?:everyone|everybody|my family|them|others|all of you)\\b",
  "\\b(?:don'?t|dont|do not) want to wake up\\b",
  "\\b(?:don'?t|dont|can'?t|cant) see (?:the|any) (?:point|reason) (?:in|to|of) (?:living|life|going on|go on)\\b",
  "\\bmarna chah(?:ta|ti) h(?:u|oon|un)\\b",
  "\\bjeene ka (?:mann|man) nahi\\b",
  "\\baatm ?hatya\\b",
  "आत्महत्या",
  "मरना चाहत",
];

export const WATCH_PATTERNS = [
  "\\bno point (?:in|to) (?:living|going on|anything)\\b",
  "\\bcan'?t (?:do|take) this anymore\\b",
  "\\bcannot (?:do|take) this anymore\\b",
  "\\bgive up on (?:life|everything)\\b",
  "\\bhopeless\\b",
  "\\bwant to disappear forever\\b",
  "\\bdisappear forever\\b",
  "\\bnot wake up\\b",
  "\\b(?:don'?t|dont|can'?t|cant) see (?:the|any) (?:point|reason) (?:in|to|of) anything\\b",
];

const CRISIS_RES = CRISIS_PATTERNS.map((p) => new RegExp(p, "i"));
const WATCH_RES = WATCH_PATTERNS.map((p) => new RegExp(p, "i"));

export const CRISIS_MESSAGE =
  "What is written here sounds like it may involve thoughts of ending life or hurting oneself. " +
  "This tool is not built for that, so it is stepping aside instead of responding.\n\n" +
  "Support from a trained person is available at any hour:\n" +
  "- India: Tele-MANAS 14416 or 1-800-891-4416 (free, 24x7, many languages)\n" +
  "- US: call or text 988\n" +
  "- UK and Ireland: Samaritans 116 123\n" +
  "- Elsewhere: findahelpline.com lists lines by country\n\n" +
  "If there is immediate danger, the local emergency number reaches help fastest. " +
  "Contacting one trusted person who can be physically nearby is also an option. " +
  "No reply to this message is needed.";

export const WATCH_NOTE =
  "A note, offered once, requiring nothing: if thoughts of not wanting to be alive are present, " +
  "trained support is available at any hour. India: Tele-MANAS 14416. US: 988. UK and Ireland: " +
  "Samaritans 116 123. Elsewhere: findahelpline.com.";

const ZERO_WIDTH = /[\u200b\u200c\u200d\u2060\ufeff]/g;
const QUOTES = /['\u2019\u2018`]/g;
const SPACED = /(?:\b\w\s+){3,}\w\b/g;
const PUNCT_COLLAPSE = /(?<=\b\w)[.\-_/](?=\w\b)/g;

export function normalize(text) {
  let t = (text || "").normalize("NFKC").toLowerCase();
  t = t.replace(ZERO_WIDTH, "");
  t = t.replace(PUNCT_COLLAPSE, "");
  t = t.replace(QUOTES, "'");
  t = t.replace(SPACED, (m) => m.replace(/\s+/g, ""));
  return t;
}

/** @returns {"crisis" | "watch" | "none"} */
export function assessRisk(text) {
  const t = normalize(text);
  if (CRISIS_RES.some((re) => re.test(t))) return "crisis";
  if (WATCH_RES.some((re) => re.test(t))) return "watch";
  return "none";
}
