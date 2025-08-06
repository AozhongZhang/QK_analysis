from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"  # or "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

model.eval()

# 第一步：准备输入
input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)

# 前向传播 + 获取 KV cache
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values

# 查看 KV cache 结构（每一层）
print(f"KV cache 包含 {len(past_key_values)} 层")
print(f"每层 key 的 shape: {past_key_values[0][0].shape}")  # (batch, num_heads, seq_len, head_dim)
print(f"每层 value 的 shape: {past_key_values[0][1].shape}")
# print(past_key_values)

# # 第二步：继续生成，使用 past_key_values
next_input = tokenizer(" Paris", return_tensors="pt", add_special_tokens=False).to(model.device)
print(next_input)
next_input["input_ids"] = next_input["input_ids"][:, -1:]
next_input["attention_mask"] = next_input["attention_mask"][:, -1:]
print(next_input)
with torch.no_grad():
    next_outputs = model(**next_input, past_key_values=past_key_values, use_cache=True)
    # next_logits = next_outputs.logits
    # updated_kv_cache = next_outputs.past_key_values
    print("第二段 logits shape:", next_outputs.logits.shape)
    print("更新后的第一层 KV key shape:", next_outputs.past_key_values[0][0].shape) 
