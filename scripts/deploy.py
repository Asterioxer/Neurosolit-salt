"""
Deploy all three Spaces to Hugging Face with the Python API (works on Windows,
no git or bash needed).

    python scripts/deploy.py --hf-username YOU
    python scripts/deploy.py --hf-username YOU --only arch1,arch3
    python scripts/deploy.py --hf-username YOU --onnx-repo YOU/nd-solitude-qwen-onnx \
        --merged-repo YOU/nd-solitude-qwen-1.5b-merged

Auth: run `hf auth login` once (or set HF_TOKEN). The token needs Write access.

Free-tier behavior:
  arch1  static Space                       -> free
  arch2  ZeroGPU (app.py adapts to CPU automatically if ZeroGPU can't be created)
  arch3  ZeroGPU (app.py adapts to CPU automatically if ZeroGPU can't be created)

  Free accounts get 2 ZeroGPU Spaces. arch2 + arch3 uses both slots. Plain
  CPU "gradio" Spaces require a PRO subscription on Hugging Face's current
  pricing -- that's the 402 Payment Required error if you ever see it.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "node_modules")


def stage(src: Path) -> Path:
    dst = Path(tempfile.mkdtemp(prefix="neuro-stage-")) / src.name
    shutil.copytree(src, dst, ignore=IGNORE)
    return dst


def deploy_space(api, repo_id, folder, sdk, hardware=None, variables=None, dry_run=False):
    print(f"\n== {repo_id} ({sdk}{', ' + hardware if hardware else ''})")
    if dry_run:
        print(f"   [dry-run] would upload {folder}")
        return
    used_hardware = "cpu-basic" if sdk == "gradio" else None
    if api.repo_exists(repo_id, repo_type="space"):
        print("   Space already exists; skipping create_repo (its current hardware is kept).")
        used_hardware = hardware or used_hardware
    elif hardware:
        try:
            api.create_repo(repo_id, repo_type="space", space_sdk=sdk, space_hardware=hardware, exist_ok=True)
            used_hardware = hardware
        except Exception as exc:  # noqa: BLE001
            print(f"   Could not create with hardware={hardware} ({exc}).")
            print("   Falling back to free cpu-basic; app.py adapts to CPU automatically.")
            try:
                api.create_repo(repo_id, repo_type="space", space_sdk=sdk, exist_ok=True)
                used_hardware = "cpu-basic"
            except Exception as exc2:  # noqa: BLE001
                print(f"   cpu-basic also failed ({exc2}). This Space likely needs")
                print("   a PRO subscription (https://huggingface.co/pro) or a free")
                print("   ZeroGPU slot -- free accounts get only 2.")
                raise
    else:
        api.create_repo(repo_id, repo_type="space", space_sdk=sdk, exist_ok=True)

    for key, value in (variables or {}).items():
        api.add_space_variable(repo_id, key, value)
        print(f"   variable {key}={value}")

    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="space", commit_message="Deploy")
    print(f"   hardware: {used_hardware or 'n/a (static)'}")
    print(f"   live at:  https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-username", default=None, help="Defaults to the logged-in account")
    p.add_argument("--only", default="arch1,arch2,arch3")
    p.add_argument("--onnx-repo", default=None, help="Custom ONNX model repo for arch1")
    p.add_argument("--merged-repo", default=None, help="Custom merged HF model repo for arch2/arch3")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api = HfApi()
    try:
        user = args.hf_username or api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Not logged in ({exc}). Run `hf auth login` or set HF_TOKEN, then retry.")

    wanted = {s.strip() for s in args.only.split(",")}

    if "arch1" in wanted:
        folder = stage(REPO_ROOT / "arch1_liminal_space")
        if args.onnx_repo:
            worker = folder / "worker.js"
            text = worker.read_text(encoding="utf-8")
            assert "const CUSTOM_MODEL_ID = null;" in text, "worker.js marker missing"
            worker.write_text(text.replace("const CUSTOM_MODEL_ID = null;", f'const CUSTOM_MODEL_ID = "{args.onnx_repo}";'), encoding="utf-8")
        deploy_space(api, f"{user}/liminal-space", folder, "static", dry_run=args.dry_run)

    if "arch2" in wanted:
        folder = stage(REPO_ROOT / "arch2_declarative_ally")
        variables = {"NEURO_MODEL_ID": args.merged_repo} if args.merged_repo else None
        deploy_space(api, f"{user}/declarative-ally", folder, "gradio", hardware="zero-a10g", variables=variables, dry_run=args.dry_run)

    if "arch3" in wanted:
        folder = stage(REPO_ROOT / "arch3_neuroact_solitude")
        variables = {"NEURO_MODEL_ID": args.merged_repo} if args.merged_repo else None
        deploy_space(api, f"{user}/neuroact-solitude", folder, "gradio", hardware="zero-a10g", variables=variables, dry_run=args.dry_run)

    print("\nDone. First builds take a few minutes.")
    print(f"Then verify: python scripts/smoke_test.py --hf-username {user}")


if __name__ == "__main__":
    main()
