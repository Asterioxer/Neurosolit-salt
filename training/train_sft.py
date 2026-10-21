"""
Stage 1: LoRA SFT on the `chosen` responses of the DPO pair file.

Why: DPO alone only re-ranks what the base model already generates. The
declarative style in `chosen` is far from Qwen's default output, so DPO
never taught it -- it just pushed long hedgy replies down until the model
collapsed to "[Pause]" / "Understood.". SFT teaches the style directly.

Output:
    ./nd_sft_adapter        LoRA adapter
    ./nd_sft_merged_local   merged fp16 model -> use as --base-model for train_dpo.py

Usage (small GPU):
    python training/train_sft.py --dataset-file data/dpo_pairs_mpingale_v2.jsonl \
        --batch-size 1 --grad-accum 8 --max-length 256 --gradient-checkpointing
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

SYSTEM_PROMPT = (
    "Act as a low-demand, neurodiversity-affirming clinical companion grounded in "
    "Linda K. Murphy's BERMED Declarative Language, Richard Erskine's Schizoid-Process "
    "Attunement to Silence, and Johan Pahnke's NeuroACT. Never ask questions, never give "
    "imperative commands, and never pressure the user to socialize."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-file", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--adapter-dir", default="./nd_sft_adapter")
    p.add_argument("--merged-dir", default="./nd_sft_merged_local")
    p.add_argument("--ckpt-dir", default="./nd_sft_checkpoints")
    p.add_argument("--no-merge", action="store_true", help="save the LoRA adapter only (use for 7B on small-RAM machines)")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("json", data_files=args.dataset_file, split="train")

    # prompt/completion format: loss is computed on the completion only.
    def fmt(ex):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            "completion": ex["chosen"] + tokenizer.eos_token,
        }

    train = raw.map(fmt, remove_columns=raw.column_names)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb, device_map="auto")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.ckpt_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        completion_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    if args.no_merge:
        print(f"Adapter saved to {args.adapter_dir} (merge skipped)")
        return

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_fp16 = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base_fp16, args.adapter_dir).merge_and_unload()
    merged.save_pretrained(args.merged_dir)
    tokenizer.save_pretrained(args.merged_dir)
    print(f"SFT merged model saved to {args.merged_dir}")


if __name__ == "__main__":
    main()
