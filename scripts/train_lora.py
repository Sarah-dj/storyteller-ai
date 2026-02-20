import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def build_text(example):
    """
    Your structured.jsonl looks like:
      {"instruction": "...", "outline": [...], "story": "..."}
    We convert it into a strict SFT format so the model learns structure + endings.
    """
    instruction = (example.get("instruction") or "").strip()
    outline = example.get("outline") or []
    story = (example.get("story") or "").strip()

    outline_block = ""
    if isinstance(outline, list) and len(outline) > 0:
        outline_block = "\n".join([f"- {x}" for x in outline if str(x).strip()])
        outline_block = f"\n### Outline:\n{outline_block}\n"

    # Strong training template (this matters more than any decoding tweak)
    text = (
        "### Instruction:\n"
        f"{instruction}\n"
        f"{outline_block}"
        "\n### Story:\n"
        f"{story}\n"
    )
    return {"text": text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/structured.jsonl")
    ap.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out_dir", type=str, default="outputs/lora_full_2m_v2")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load dataset ----------
    ds = load_dataset("json", data_files=args.data_path, split="train")

    # Ruthless filtering: remove empty / ultra-short stories (they teach the model to end early)
    def good_example(ex):
        story = (ex.get("story") or "").strip()
        return len(story) >= 600  # chars (~200+ tokens often). Raise/lower if needed.

    ds = ds.filter(good_example)

    # Build strong formatted training text
    ds = ds.map(build_text, remove_columns=ds.column_names)

    # Train/eval split
    splits = ds.train_test_split(test_size=0.05, seed=args.seed)
    train_ds, eval_ds = splits["train"], splits["test"]

    # ---------- 4-bit load (Colab T4/L4 friendly) ----------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto"
    )

    # ---------- Strong LoRA (attention + MLP) ----------
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    model.train()

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16

    # ---------- Training ----------
    targs = TrainingArguments(
        output_dir=args.out_dir,

        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,

        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,

        logging_steps=50,

        evaluation_strategy="steps",
        eval_steps=200,

        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="paged_adamw_8bit",

        fp16=fp16,
        bf16=bf16,

        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        packing=True,   # BIG upgrade: uses context efficiently + improves structure learning
        args=targs,
    )

    trainer.train()

    # ---------- Save adapter + tokenizer ----------
    adapter_dir = os.path.join(args.out_dir, "adapter")
    tok_dir = os.path.join(args.out_dir, "tokenizer")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(tok_dir)

    print(f"✅ Saved adapter: {adapter_dir}")
    print(f"✅ Saved tokenizer: {tok_dir}")

if __name__ == "__main__":
    main()