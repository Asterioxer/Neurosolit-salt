"""
One command from a fresh clone to three live Spaces (stock models) on the free tier.

    pip install -r requirements-dev.txt
    hf auth login            # once, WRITE token
    python run_all.py --hf-username YOU

Stages (each can be skipped with --skip tests,dataset,deploy,smoke):
  tests    safety-layer tests + check that the three safety copies are in sync
  dataset  builds data/dpo_pairs.jsonl (also pushed to the Hub with --push-dataset)
  deploy   uploads arch1, arch2, arch3 (arch2 falls back to free CPU if ZeroGPU is unavailable)
  smoke    waits for the Spaces to build, then checks crisis gate + no '?' in output

Training a custom model is a separate GPU step: training/colab_pipeline.ipynb.
After it finishes, re-run `python scripts/deploy.py` with the model flags it prints.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, check=True):
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check).returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-username", required=True)
    p.add_argument("--skip", default="", help="comma list: tests,dataset,deploy,smoke")
    p.add_argument("--push-dataset", action="store_true")
    p.add_argument("--build-wait", type=int, default=600, help="seconds to wait for Space builds before smoke tests")
    args = p.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    py = sys.executable

    if "tests" not in skip:
        run([py, "scripts/check_sync.py"])
        run([py, "-m", "pytest", "tests", "-q"])
    if "dataset" not in skip:
        cmd = [py, "dataset/build_dpo_dataset.py", "--hf-username", args.hf_username]
        if args.push_dataset:
            cmd.append("--push")
        run(cmd)
    if "deploy" not in skip:
        run([py, "scripts/deploy.py", "--hf-username", args.hf_username])
    if "smoke" not in skip:
        print(f"\nWaiting {args.build_wait}s for Space builds (arch3 also downloads a ~1 GB model)...")
        time.sleep(args.build_wait)
        rc = run([py, "scripts/smoke_test.py", "--hf-username", args.hf_username], check=False)
        if rc != 0:
            print("Smoke test failed or Spaces are still building. Check build logs on the Space pages, "
                  "then re-run: python scripts/smoke_test.py --hf-username " + args.hf_username)


if __name__ == "__main__":
    main()
