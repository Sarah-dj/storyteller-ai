import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


# ----------------------------
# STEP 1 — Model name
# ----------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model:", MODEL_NAME)


# ----------------------------
# STEP 2 — Load tokenizer + model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

print("Training device:", device)


# ----------------------------
# STEP 3 — Attach LoRA adapters
# ----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
print("LoRA ready.")


# ----------------------------
# STEP 4 — Load your dataset
# ----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train/train.jsonl",
        "validation": "data/train/val.jsonl",
    }
)

print(dataset)


# ----------------------------
# STEP 5 — Tokenization
# ----------------------------
MAX_LEN = 512

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

tok_train = dataset["train"].map(tokenize)
tok_val = dataset["validation"].map(tokenize)


# ----------------------------
# STEP 6 — Collator
# ----------------------------
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ----------------------------
# STEP 7 — Training arguments
args = TrainingArguments(
    output_dir="models/adapters/tinyllama-lora",

    # training
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,

    # logging
    logging_steps=1,
    save_steps=50,

    # IMPORTANT (Mac stability)
    fp16=False,
    bf16=False,

    # disable evaluation completely for now
    do_eval=False,

    report_to="none"
)


# ----------------------------
# STEP 8 — Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=collator,
)

print("Starting training...")

trainer.train()


# ----------------------------
# STEP 9 — Save adapter
# ----------------------------
trainer.model.save_pretrained("models/adapters/tinyllama-lora")
tokenizer.save_pretrained("models/adapters/tinyllama-lora")

print("Training complete. Adapter saved.")
