import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter_dir", type=str, default="outputs/lora_full_2m_v2/adapter")
    ap.add_argument("--max_new_tokens", type=int, default=520)
    args = ap.parse_args()

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

    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    prompt = """### Instruction:
Write a COMPLETE children's story (220–350 words) with:
1) a clear beginning
2) a problem or mystery
3) an unexpected moment
4) a satisfying ending (no sudden cut)

Theme: Tommy discovers something strange in the garden.

### Story:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=220,
            do_sample=True,
            temperature=0.95,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    print(full[len(prompt):].strip())

if __name__ == "__main__":
    main()