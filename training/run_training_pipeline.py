"""
Full training pipeline in one command (needs a CUDA GPU and Linux for the GGUF step;
free Colab T4 works -- see training/colab_pipeline.ipynb).

    python training/run_training_pipeline.py --hf-username YOU

Stages: build dataset -> QLoRA DPO train + merge + push -> ONNX export + push
        -> GGUF export + push. Prints the exact deploy command at the end.
Skip stages with --skip dataset,train,onnx,gguf.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-username", required=True)
    p.add_argument("--skip", default="", help="comma list: dataset,train,onnx,gguf")
    args = p.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    u = args.hf_username
    merged = f"{u}/nd-solitude-qwen-1.5b-merged"

    if "train" not in skip:
        import torch

        if not torch.cuda.is_available():
            sys.exit("No CUDA GPU found. Run this on Colab/Kaggle (training/colab_pipeline.ipynb), or --skip train.")

    if "dataset" not in skip:
        run([sys.executable, "dataset/build_dpo_dataset.py", "--hf-username", u, "--push"])
    if "train" not in skip:
        run([sys.executable, "training/train_dpo.py", "--hf-username", u,
             "--dataset-file", "data/dpo_pairs.jsonl", "--push"])
    if "onnx" not in skip:
        run([sys.executable, "training/export_onnx.py", "--source-model", merged, "--hf-username", u, "--push"])
    if "gguf" not in skip:
        run(["bash", "training/export_gguf.sh", u, merged])

    print("\nPipeline finished. Deploy with your models:")
    print(f"  python scripts/deploy.py --hf-username {u} \\")
    print(f"    --onnx-repo {u}/nd-solitude-qwen-onnx --merged-repo {merged} --gguf-repo {u}/nd-solitude-qwen-gguf")
    print("Before sharing publicly: evaluate the merged model on held-out prompts (see docs/ROADMAP.md Phase 2).")


if __name__ == "__main__":
    main()
