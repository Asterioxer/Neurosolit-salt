"""
QLoRA SFT-style preparation + span-aware DPO alignment.

Designed for a free Google Colab T4 GPU or Kaggle GPU notebook. Can also run
on a small local GPU (e.g. a 4GB laptop card) with the --batch-size,
--grad-accum, --max-length, and --max-prompt-length flags turned down -- the
defaults below assume a larger GPU (T4, 16GB) and will OOM on 4GB cards.

    pip install -U torch transformers datasets trl peft bitsandbytes accelerate huggingface_hub
    python train_dpo.py --hf-username YOUR_USERNAME
    python train_dpo.py --hf-username YOUR_USERNAME --batch-size 1 --grad-accum 8 --max-length 256 --max-prompt-length 128   # small/4GB GPU

This trains the model to prefer declarative, low-demand responses (chosen)
over imperative/interrogative ones (rejected), with an additional penalty
weight applied via `--span-penalty-weight` against residual question marks
and imperative-verb openers in generation during evaluation (penalty is
applied as a generation-time scoring aid for manual review; the core
optimization is standard DPOTrainer on the practitioner-labeled pairs).
"""

import argparse
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

SYSTEM_PROMPT = (
    "Act as a low-demand, neurodiversity-affirming clinical companion grounded in "
    "Linda K. Murphy's BERMED Declarative Language, Richard Erskine's Schizoid-Process "
    "Attunement to Silence, and Johan Pahnke's NeuroACT. Never ask questions, never give "
    "imperative commands, and never pressure the user to socialize."
)

_IMPERATIVE_PATTERN = re.compile(
    r"\b(you should|you need to|you must|try to|start by|make sure)\b", re.IGNORECASE
)


def span_penalty_score(text: str) -> int:
    """
    Counts residual '?' characters and imperative-verb openers. Used for
    manual post-training evaluation (see `--eval-after-train`), not as part
    of the loss function itself -- TRL's DPOTrainer optimizes on the
    provided chosen/rejected pairs directly. If you want the penalty to
    affect the loss, filter/re-weight the dataset upstream so
    `chosen` rows already score 0 here, rather than modifying DPOTrainer's
    internals.
    """
    return text.count("?") + len(_IMPERATIVE_PATTERN.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-username", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset-repo", default=None, help="Defaults to <username>/nd-solitude-dpo-pairs")
    parser.add_argument("--dataset-file", default=None, help="Local JSONL from build_dpo_dataset.py (takes priority over --dataset-repo)")
    parser.add_argument("--merged-repo", default=None, help="Defaults to <username>/nd-solitude-qwen-1.5b-merged")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL constraint strength")
    parser.add_argument("--batch-size", type=int, default=2, help="per_device_train_batch_size; lower uses less VRAM")
    parser.add_argument("--grad-accum", type=int, default=4, help="gradient_accumulation_steps; raise this as --batch-size drops to keep the same effective batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max total sequence length (prompt+response); lower uses less VRAM")
    parser.add_argument("--max-prompt-length", type=int, default=256, help="Max prompt length; lower uses less VRAM")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Trade compute for VRAM; recommended on small (<=6GB) GPUs")
    parser.add_argument("--rpo-alpha", type=float, default=None, help="Adds an NLL (SFT) term on chosen responses to the DPO loss; stops chosen likelihood drifting down. 1.0 is a reasonable start.")
    parser.add_argument("--adapter-dir", default="./nd_dpo_adapter")
    parser.add_argument("--merged-dir", default="./nd_dpo_merged_local")
    parser.add_argument("--ckpt-dir", default="./nd_dpo_checkpoints")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    dataset_repo = args.dataset_repo or f"{args.hf_username}/nd-solitude-dpo-pairs"
    merged_repo = args.merged_repo or f"{args.hf_username}/nd-solitude-qwen-1.5b-merged"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset_file:
        raw_dataset = load_dataset("json", data_files=args.dataset_file, split="train")
    else:
        raw_dataset = load_dataset(dataset_repo, split="train")
    if len(raw_dataset) < 100:
        print(f"WARNING: only {len(raw_dataset)} pairs; expect a weak model.")

    # Sanity check: warn (don't crash) if a chosen response still scores
    # high on the span penalty -- this usually means the dataset needs
    # another review pass before training.
    flagged = sum(1 for r in raw_dataset if span_penalty_score(r["chosen"]) > 0)
    if flagged:
        print(
            f"WARNING: {flagged}/{len(raw_dataset)} 'chosen' rows still contain "
            "question marks or imperative openers. Review dataset/build_dpo_dataset.py "
            "output before training on it."
        )

    def format_row(example):
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True),
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }

    train_dataset = raw_dataset.map(format_row, remove_columns=raw_dataset.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = DPOConfig(
        output_dir=args.ckpt_dir,
        rpo_alpha=args.rpo_alpha,
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.model.save_pretrained(args.adapter_dir)

    # Free the quantized training model and its optimizer state before
    # reloading the base model for merging -- on small GPUs this avoids a
    # second OOM at merge time (two full model copies otherwise coexist).
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reload base model in FP16, merge LoRA weights
    base_fp16 = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    merged_model = PeftModel.from_pretrained(base_fp16, args.adapter_dir)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(args.merged_dir)
    tokenizer.save_pretrained(args.merged_dir)

    print(f"Merged model saved locally to {args.merged_dir}")
    print(
        "IMPORTANT: evaluate the merged model against held-out prompts before "
        "pushing or deploying -- check for both guardrail violations (residual "
        "'?' / imperatives) AND for the opposite failure mode, where the model "
        "becomes unhelpfully evasive or refuses to engage at all."
    )

    if args.push:
        merged_model.push_to_hub(merged_repo)
        tokenizer.push_to_hub(merged_repo)
        print(f"Pushed to https://huggingface.co/{merged_repo}")


if __name__ == "__main__":
    main()
