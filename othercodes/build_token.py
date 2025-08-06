"""
build_prompts_llama3.py
-------------------------------------------------
从 wikitext-2 数据集中连续抽取文本，
用 Llama-3-8B 的 tokenizer 生成 1k~128k token 的多档 prompt，
输出 JSON：
{
  "1k": "...",
  "2k": "...",
  ...
  "128k": "..."
}
"""

import os, json
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------- 配置 ----------
# MODEL_ID =  "meta-llama/Meta-Llama-3.1-8B-Instruct"   # Llama-3-8B
MODEL_ID = 'meta-llama/Llama-2-7b-hf'
TARGET_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]  # 1k~128k
OUT_DIR = "output"
OUT_FILE = "wikitext2_prompts_llama2.json"

# ---------- 初始化 tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=True,
)

# ---------- 加载 wikitext-2 ----------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
texts = dataset["train"]["text"]

# ---------- 构造 prompt ----------
prompts = {}
idx = 0  # 游标
for tgt in TARGET_LENS:
    prompt, tok_cnt = "", 0
    while tok_cnt < tgt and idx < len(texts):
        line = texts[idx].strip()
        idx += 1
        if not line:
            continue
        candidate = prompt + line + "\n"
        if len(tokenizer(candidate)["input_ids"]) <= tgt:
            prompt = candidate
            tok_cnt = len(tokenizer(prompt)["input_ids"])
        else:
            break
    key = f"{tgt // 1024}k"
    prompts[key] = prompt.strip()
    print(f"[✔] {key} done: {tok_cnt} tokens.")

# ---------- 保存 ----------
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, OUT_FILE), "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)

print(f"\n✅ All prompts saved to {os.path.join(OUT_DIR, OUT_FILE)}")
