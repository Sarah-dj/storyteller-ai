import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ✅ Base model
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ✅ Your trained adapter path
ADAPTER_PATH = "models/adapters/tinyllama-lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("\n✅ Model ready!\n")

# ---------------------------------------------------
# Prompt test
# ---------------------------------------------------

prompt = """Write a fantasy story.

TITLE:
SETTING: forest
TONE: mysterious
LEVEL: 7

STORY:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...\n")

output = model.generate(
    **inputs,
    max_new_tokens=250,
    temperature=0.9,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))