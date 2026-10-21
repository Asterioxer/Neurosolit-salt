#!/usr/bin/env bash
# Quantize a merged HF model to Q4_K_M GGUF for Architecture 3
# (NeuroACT-Solitude, CPU-Basic Space via llama-cpp-python).
#
# Usage:
#   ./export_gguf.sh <hf-username> <source-model-repo>
#
# Run on Linux (free Colab works). Needs git, cmake, and a logged-in HF token
# (HF_TOKEN env var or `hf auth login`). Uses the Python API for Hub
# operations so it does not depend on CLI subcommand names.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <hf-username> <source-model-repo>"
  exit 1
fi

HF_USERNAME="$1"
SOURCE_MODEL="$2"
GGUF_REPO="${HF_USERNAME}/nd-solitude-qwen-gguf"

echo "1. Cloning and building llama.cpp..."
if [ ! -d "llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp
cmake -B build && cmake --build build --config Release -j 4 --target llama-quantize
pip install -r requirements.txt
cd ..

echo "2. Downloading merged model: ${SOURCE_MODEL}..."
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${SOURCE_MODEL}", local_dir="./merged_model")
PY

echo "3. Converting to FP16 GGUF..."
python llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile nd-solitude-f16.gguf --outtype f16

echo "4. Quantizing to Q4_K_M..."
./llama.cpp/build/bin/llama-quantize nd-solitude-f16.gguf nd-solitude-Q4_K_M.gguf Q4_K_M

echo "5. Uploading to ${GGUF_REPO}..."
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("${GGUF_REPO}", repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="nd-solitude-Q4_K_M.gguf", path_in_repo="nd-solitude-Q4_K_M.gguf", repo_id="${GGUF_REPO}", repo_type="model")
PY

echo "Done. Deploy with:  python scripts/deploy.py --hf-username ${HF_USERNAME} --gguf-repo ${GGUF_REPO}"
