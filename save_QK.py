# Get the output of Q/K for 4k and 64K before rope and after rope
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from types import MethodType
import json


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

target_length_4k = "64k"

prompt_4k = prompts[target_length_4k]
inputs_4k = tokenizer(prompt_4k, return_tensors="pt", add_special_tokens=False).to(model.device)
seq_len_4 = inputs_4k["input_ids"].shape[1]
print(seq_len_4)

# prompt = "Hi! I just wanted to see how everything is going with you. It's been a while since we last talked. I hope you're doing well."
# input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
# seq_length = input["input_ids"].shape[1]
# print(seq_length)

cache_64k = {}
target_layers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

def make_patched_forward(layer_idx):
    def patched_forward(self, hidden_states, position_embeddings=None, *args, **kwargs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        bsz, seqlen, dim = q.shape
        head_dim = self.head_dim
        num_heads_q = self.config.num_attention_heads
        num_heads_kv = self.config.num_key_value_heads

        q = q.view(bsz, seqlen, num_heads_q, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

        cache_64k[layer_idx] = {
                # "q_raw": q.detach().cpu(),
                "k_raw": k.detach().cpu(),
                # "q_rope": q_rope.detach().cpu(),
                "k_rope": k_rope.detach().cpu(),
            }

        return self._orig_forward(hidden_states, position_embeddings, *args, **kwargs)
    
    return patched_forward

for layer_idx in target_layers:
    attn_layer = model.model.layers[layer_idx].self_attn
    attn_layer._orig_forward = attn_layer.forward  # 保存原始 forward
    attn_layer.forward = MethodType(make_patched_forward(layer_idx), attn_layer)


with torch.no_grad():
    outputs = model(**inputs_4k)

torch.save(cache_64k, "/home/azzhang/QK_analysis/cache_64k.pt")