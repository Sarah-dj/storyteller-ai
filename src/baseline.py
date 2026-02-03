from transformers import pipeline

generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt = "Write a humorous fantasy quest in the mountains."

out = generator(prompt, max_new_tokens=200)

print(out[0]["generated_text"])

