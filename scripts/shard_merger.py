"""
Low-memory, shard-by-shard LoRA merger for Qwen2.5-7B.
Fuses LoRA weights directly into safetensors shards without loading the entire
15 GB model into memory. Peak RAM usage is < 4 GB (fits easily on standard Colab CPU).
"""

import gc
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_file, save_file
import torch


def merge_lora_into_shards(
    base_repo: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_source: str = "pipeline_out/adapter",
    output_dir: str = "pipeline_out/merged",
    push_to_repo: str = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Resolve adapter files (local folder or Hugging Face repo)
    adapter_path = Path(adapter_source)
    if adapter_path.is_dir() and (adapter_path / "adapter_model.safetensors").exists():
        print(f"1. Loading adapter weights from local directory: {adapter_source}...")
        adapter_weights_file = str(adapter_path / "adapter_model.safetensors")
        adapter_config_file = str(adapter_path / "adapter_config.json")
    else:
        print(f"1. Downloading adapter files from Hugging Face: {adapter_source}...")
        adapter_weights_file = hf_hub_download(adapter_source, "adapter_model.safetensors")
        adapter_config_file = hf_hub_download(adapter_source, "adapter_config.json")

    with open(adapter_config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    alpha = config.get("lora_alpha", 32)
    r = config.get("r", 16)
    scale = alpha / r
    print(f"   LoRA scaling factor: alpha={alpha}, r={r} -> scale={scale}")

    adapter_tensors = load_file(adapter_weights_file)
    print(f"   Loaded {len(adapter_tensors)} adapter weight tensors.")

    # Index adapter tensors by target module base key
    lora_pairs = {}
    for k in adapter_tensors.keys():
        if ".lora_A.weight" in k:
            base_key = k.replace("base_model.model.", "").replace(".lora_A.weight", "")
            b_key = k.replace(".lora_A.weight", ".lora_B.weight")
            if b_key in adapter_tensors:
                lora_pairs[base_key + ".weight"] = (k, b_key)

    print(f"   Found {len(lora_pairs)} modules with LoRA adapters to merge.")

    # 2. Get shard index from base model
    print(f"2. Fetching shard index from base model: {base_repo}...")
    index_file = hf_hub_download(base_repo, "model.safetensors.index.json")
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    shard_files = sorted(list(set(index_data["weight_map"].values())))
    print(f"   Base model has {len(shard_files)} shards to process.")

    # Copy configs and tokenizer assets
    config_files = [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"
    ]
    api = HfApi()
    for cf in config_files:
        try:
            if adapter_path.is_dir() and (adapter_path / cf).exists():
                (out_path / cf).write_bytes((adapter_path / cf).read_bytes())
            else:
                try:
                    cf_path = hf_hub_download(adapter_source, cf)
                except Exception:
                    cf_path = hf_hub_download(base_repo, cf)
                (out_path / cf).write_bytes(Path(cf_path).read_bytes())
        except Exception:
            pass

    (out_path / "model.safetensors.index.json").write_bytes(Path(index_file).read_bytes())

    # 3. Process each shard sequentially
    total_merged_weights = 0
    for idx, shard_name in enumerate(shard_files, 1):
        print(f"\n3.{idx} Downloading & merging shard: {shard_name}...")
        shard_path = hf_hub_download(base_repo, shard_name)
        shard_tensors = load_file(shard_path)

        merged_count = 0
        for weight_name, (a_key, b_key) in lora_pairs.items():
            if weight_name in shard_tensors:
                base_w = shard_tensors[weight_name].float()
                lora_a = adapter_tensors[a_key].float()
                lora_b = adapter_tensors[b_key].float()
                delta = torch.matmul(lora_b, lora_a) * scale
                shard_tensors[weight_name] = (base_w + delta).to(shard_tensors[weight_name].dtype)
                merged_count += 1
                total_merged_weights += 1

        print(f"   Merged {merged_count} target modules in this shard.")
        dst_shard = out_path / shard_name
        print(f"   Writing merged shard: {dst_shard.name}...")
        save_file(shard_tensors, str(dst_shard))

        del shard_tensors
        gc.collect()

    print(f"\nSUCCESS: All {total_merged_weights} layers fused across {len(shard_files)} shards!")
    print(f"Standalone merged model is ready at: {out_path}")

    if push_to_repo:
        print(f"\n4. Uploading standalone 7B merged model to {push_to_repo}...")
        api.create_repo(repo_id=push_to_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(out_path),
            repo_id=push_to_repo,
            commit_message="Add standalone merged Qwen2.5-7B low-demand solitude model"
        )
        print(f"Model successfully published to: https://huggingface.co/{push_to_repo}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-repo", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="Scadiva/nd-solitude-7b-adapter")
    parser.add_argument("--output-dir", default="pipeline_out/merged")
    parser.add_argument("--push", default=None)
    args = parser.parse_args()

    merge_lora_into_shards(
        base_repo=args.base_repo,
        adapter_source=args.adapter,
        output_dir=args.output_dir,
        push_to_repo=args.push
    )
