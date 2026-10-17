#!/usr/bin/env bash
# Quantize a merged HF model to Q4_K_M GGUF for Architecture 3
# (NeuroACT-Solitude, CPU-Basic Space via llama-cpp-python).
#
# Usage:
#   ./export_gguf.sh <hf-username> <source-model-repo>
#
# Example:
#   ./export_gguf.sh myuser myuser/nd-solitude-qwen-1.5b-merged
#
# Run this in a free Colab notebook or any machine with enough disk/RAM to
# hold the fp16 model plus a quantized copy simultaneously.

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
cmake -B build && cmake --build build --config Release -j 4
pip install -r requirements.txt
cd ..

echo "2. Downloading merged model: ${SOURCE_MODEL}..."
hf download "${SOURCE_MODEL}" --local-dir ./merged_model

echo "3. Converting to FP16 GGUF..."
python llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile nd-solitude-f16.gguf --outtype f16

echo "4. Quantizing to Q4_K_M..."
./llama.cpp/build/bin/llama-quantize nd-solitude-f16.gguf nd-solitude-Q4_K_M.gguf Q4_K_M

echo "5. Uploading to ${GGUF_REPO}..."
hf repos create "${GGUF_REPO}" --type model --public --exist-ok
hf upload "${GGUF_REPO}" ./nd-solitude-Q4_K_M.gguf nd-solitude-Q4_K_M.gguf

echo "Done. Point arch3_neuroact_solitude/app.py GGUF_REPO/GGUF_FILE at:"
echo "  repo: ${GGUF_REPO}"
echo "  file: nd-solitude-Q4_K_M.gguf"
