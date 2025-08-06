import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# ✅ 设置参数
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = 'meta-llama/Llama-2-7b-hf'
token_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
max_samples = 10_000
OUT_DIR = "output"
output_json_name = "codeparrot_prompts_llama2.json"

# ✅ 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 加载数据集
print("🔹 Loading dataset...")
dataset = load_dataset("codeparrot/github-jupyter-code-to-text", split="train")

# ✅ 拼接 code 块直到达到最大 token 要求
print("🔹 Building long code string...")
code_blocks = []
token_count = 0
sample_count = 0

for example in dataset:
    code = example.get("content")
    if not code or not code.strip():
        continue

    code_blocks.append(code.strip() + "\n\n")
    total_text = "".join(code_blocks)
    token_count = len(tokenizer(total_text, return_attention_mask=False)["input_ids"])

    sample_count += 1
    if token_count > max(token_lengths) + 512 or sample_count >= max_samples:
        break

print(f"  → Total tokens collected: {token_count}")
print(f"  → Total samples used: {sample_count}")

# ✅ 构造 prompts
print("🔹 Tokenizing full code block...")
tokenized = tokenizer("".join(code_blocks), return_attention_mask=False)
input_ids = tokenized["input_ids"]

prompts = {}
for length in token_lengths:
    if len(input_ids) >= length:
        sub_ids = input_ids[:length]
        prompt = tokenizer.decode(sub_ids, skip_special_tokens=True)
        prompts[f"{length // 1024}k"] = prompt
    else:
        print(f"⚠️ Warning: insufficient tokens for {length} tokens")

# ✅ 保存为 JSON
os.makedirs(OUT_DIR, exist_ok=True)
output_path = os.path.join(OUT_DIR, output_json_name)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)

print(f"✅ Saved to: {output_path}")
