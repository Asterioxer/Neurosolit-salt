#!/usr/bin/env bash
# Provisions and pushes all three Spaces at once. Requires `hf auth login`
# to already be done. Run from the repo root.
#
# Usage: ./scripts/deploy_all_free_spaces.sh <hf-username>

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <hf-username>"
  exit 1
fi

HF_USER="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d)"

echo "Working in ${WORKDIR}"

echo "1. Provisioning Architecture 1: Liminal-Space (Static WebGPU)..."
hf repos create "${HF_USER}/liminal-space" --type space --space-sdk static --public --exist-ok
git clone "https://huggingface.co/spaces/${HF_USER}/liminal-space" "${WORKDIR}/liminal-space"
cp "${REPO_ROOT}"/arch1_liminal_space/* "${WORKDIR}/liminal-space/"
(cd "${WORKDIR}/liminal-space" && git add . && git commit -m "Deploy Liminal-Space" --allow-empty && git push)

echo "2. Provisioning Architecture 2: Declarative-Ally (Free Dynamic ZeroGPU)..."
hf repos create "${HF_USER}/declarative-ally" --type space --space-sdk gradio --flavor zero-a10g --public --exist-ok
git clone "https://huggingface.co/spaces/${HF_USER}/declarative-ally" "${WORKDIR}/declarative-ally"
cp "${REPO_ROOT}"/arch2_declarative_ally/* "${WORKDIR}/declarative-ally/"
(cd "${WORKDIR}/declarative-ally" && git add . && git commit -m "Deploy Declarative-Ally" --allow-empty && git push)

echo "3. Provisioning Architecture 3: NeuroACT-Solitude (Free CPU-Basic GGUF)..."
hf repos create "${HF_USER}/neuroact-solitude" --type space --space-sdk gradio --flavor cpu-basic --public --exist-ok
git clone "https://huggingface.co/spaces/${HF_USER}/neuroact-solitude" "${WORKDIR}/neuroact-solitude"
cp "${REPO_ROOT}"/arch3_neuroact_solitude/* "${WORKDIR}/neuroact-solitude/"
(cd "${WORKDIR}/neuroact-solitude" && git add . && git commit -m "Deploy NeuroACT-Solitude" --allow-empty && git push)

echo ""
echo "All three Spaces deployed with stock/default models. Streaming build logs for declarative-ally..."
hf spaces logs "${HF_USER}/declarative-ally" --build --follow
