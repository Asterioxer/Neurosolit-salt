"""
Export a merged HF model to quantized ONNX for browser deployment
(Architecture 1 / Liminal-Space) via Transformers.js.

    pip install -U "optimum[onnxruntime]" onnx onnxruntime
    python export_onnx.py --source-model YOUR_USERNAME/nd-solitude-qwen-1.5b-merged --hf-username YOUR_USERNAME --push

If you don't have a merged model yet, you can skip this and point
arch1_liminal_space/worker.js directly at a stock model such as
onnx-community/Qwen2.5-0.5B-Instruct to get the UI working first.
"""

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--hf-username", required=True)
    parser.add_argument("--output-repo", default=None, help="Defaults to <username>/nd-solitude-qwen-onnx")
    parser.add_argument("--raw-dir", default="./onnx_raw")
    parser.add_argument("--quant-dir", default="./onnx_quantized")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    output_repo = args.output_repo or f"{args.hf_username}/nd-solitude-qwen-onnx"
    raw_dir = Path(args.raw_dir)
    quant_dir = Path(args.quant_dir)

    print(f"Exporting {args.source_model} to ONNX...")
    ort_model = ORTModelForCausalLM.from_pretrained(args.source_model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    ort_model.save_pretrained(raw_dir)
    tokenizer.save_pretrained(raw_dir)

    print("Quantizing (AVX512 VNNI, per-channel dynamic)...")
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=quant_dir, quantization_config=qconfig)
    tokenizer.save_pretrained(quant_dir)

    print(f"Quantized ONNX saved to {quant_dir}")

    if args.push:
        import shutil

        from huggingface_hub import HfApi

        # transformers.js expects weights under onnx/ and tokenizer/config at the root.
        pub_dir = Path("./onnx_publish")
        if pub_dir.exists():
            shutil.rmtree(pub_dir)
        shutil.copytree(quant_dir, pub_dir)
        (pub_dir / "onnx").mkdir(exist_ok=True)
        for f in list(pub_dir.glob("*.onnx")) + list(pub_dir.glob("*.onnx_data")):
            shutil.move(str(f), str(pub_dir / "onnx" / f.name))

        api = HfApi()
        api.create_repo(repo_id=output_repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(pub_dir), repo_id=output_repo, repo_type="model")
        print(f"Pushed to https://huggingface.co/{output_repo}")
        print(f"Deploy with: python scripts/deploy.py --hf-username {args.hf_username} --onnx-repo {output_repo}")
        print("NOTE: custom exports load with dtype q8 (model_quantized.onnx); test in the browser before sharing.")


if __name__ == "__main__":
    main()
