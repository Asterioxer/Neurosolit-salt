"""
QLoRA SFT-style preparation + span-aware DPO alignment.

Run on a free Google Colab T4 GPU or Kaggle GPU notebook. Not intended to
run locally without a CUDA GPU.

    pip install -U torch transformers datasets trl peft bitsandbytes accelerate huggingface_hub
    python train_dpo.py --hf-username YOUR_USERNAME

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-username", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset-repo", default=None, help="Defaults to <username>/nd-solitude-dpo-pairs")
    parser.add_argument("--merged-repo", default=None, help="Defaults to <username>/nd-solitude-qwen-1.5b-merged")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL constraint strength")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    dataset_repo = args.dataset_repo or f"{args.hf_username}/nd-solitude-dpo-pairs"
    merged_repo = args.merged_repo or f"{args.hf_username}/nd-solitude-qwen-1.5b-merged"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset(dataset_repo, split="train")

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
        output_dir="./nd_dpo_checkpoints",
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        max_length=512,
        max_prompt_length=256,
        remove_unused_columns=False,
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
    trainer.model.save_pretrained("./nd_dpo_adapter")

    # Reload base model in FP16, merge LoRA weights
    base_fp16 = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    merged_model = PeftModel.from_pretrained(base_fp16, "./nd_dpo_adapter")
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained("./nd_dpo_merged_local")
    tokenizer.save_pretrained("./nd_dpo_merged_local")

    print("Merged model saved locally to ./nd_dpo_merged_local")
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
