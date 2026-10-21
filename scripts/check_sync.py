"""
Verifies the safety layer copies have not drifted apart:
  - arch2/safety.py and arch3/safety.py are byte-identical
  - arch1/crisis.js has exactly the same pattern lists as safety.py

    python scripts/check_sync.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY2 = ROOT / "arch2_declarative_ally" / "safety.py"
PY3 = ROOT / "arch3_neuroact_solitude" / "safety.py"
JS = ROOT / "arch1_liminal_space" / "crisis.js"


def js_list(src: str, name: str):
    m = re.search(rf"export const {name} = \[(.*?)\n\];", src, re.S)
    assert m, f"{name} not found in crisis.js"
    return [json.loads(s) for s in re.findall(r'^\s+("(?:[^"\\]|\\.)*"),?$', m.group(1), re.M)]


def py_list(path: Path, name: str):
    ns = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns[name]


def main() -> int:
    ok = True
    if PY2.read_bytes() != PY3.read_bytes():
        print("DRIFT: arch2 and arch3 safety.py differ")
        ok = False
    js_src = JS.read_text(encoding="utf-8")
    for name in ("CRISIS_PATTERNS", "WATCH_PATTERNS"):
        if js_list(js_src, name) != py_list(PY2, name):
            print(f"DRIFT: {name} differs between crisis.js and safety.py")
            ok = False
    print("safety layer in sync" if ok else "FIX THE DRIFT BEFORE DEPLOYING")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
