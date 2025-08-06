from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from types import MethodType
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os
import json


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
target_length = "64k"
# target_length = ["1k","2k","4k","8k","16k","32k","64k", "128k"]
# for tar_len in target_length:
prompt = prompts[target_length]
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
seq_len_1 = inputs["input_ids"].shape[1]
print(seq_len_1)

# target_layer = 4
# inputs = data_sample(256, model.device)
# for i in range(5,32):
cache = {}
i = 30

def patched_forward(self, hidden_states, *args, **kwargs):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    bsz, seqlen, dim = q.shape
    head_dim = self.head_dim
    # num_heads = self.num_heads
    num_heads_q = self.config.num_attention_heads
    num_heads_kv = self.config.num_key_value_heads
    print(num_heads_q)
    print(num_heads_kv)

    q = q.view(bsz, seqlen, num_heads_q, head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)

    # 存储RoPE之前的Q、K
    cache["q_raw"] = q.detach().cpu()
    cache["k_raw"] = k.detach().cpu()
    cache["v"] = v.detach().cpu()

    # cos, sin = self.rotary_emb(v, position_ids)  # 注意这里是 value_states 用于生成位置编码
    # q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    # cache["q_rope"] = q_rope.detach().cpu()
    # cache["k_rope"] = k_rope.detach().cpu()

    
    return self._orig_forward(hidden_states, *args, **kwargs)

# 注入 patch
attn_layer = model.model.layers[i].self_attn
attn_layer._orig_forward = attn_layer.forward
attn_layer.forward = MethodType(patched_forward, attn_layer)

with torch.no_grad():
    outputs = model(**inputs)

q_raw = cache["q_raw"].squeeze(0)  # shape: (num_heads, seq_len, head_dim)
k_raw = cache["k_raw"].squeeze(0)
v = cache["v"].squeeze(0)



def analyze_QQ_dot_product(Q: torch.Tensor, output_dir="qq_dot_product_plots", selected_heads=[0], fixed_pos_list=[0, 1000, 50000]):
    """
    Analyze QQ unnormalized dot product (Q @ Q^T) across token positions.
    
    Args:
        Q: Tensor of shape [num_heads, seq_len, head_dim]
        output_dir: Where to save plots
        selected_heads: Head indices to visualize individually
        fixed_pos_list: Token positions to visualize dot(Q[i], Q[j])
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_heads, seq_len, head_dim = Q.shape
    print(f"[Info] Q shape: {Q.shape}")

    # Do NOT normalize Q, use raw dot product
    QQ_dot = torch.matmul(Q, Q.transpose(-1, -2))  # [H, L, L]

    # # 1. Heatmap per selected head
    # for h in selected_heads:
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(QQ_dot[h].cpu(), cmap="RdBu_r")
    #     plt.title(f"QQ Dot Product (Head {h})")
    #     plt.xlabel("Token Position")
    #     plt.ylabel("Token Position")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, f"qq_dot_head{h}.png"))
    #     plt.close()

    # 2. Fixed token dot product curve
    for h in selected_heads:
        for pos in fixed_pos_list:
            sim_row = QQ_dot[h][pos]  # [seq_len]
            plt.figure(figsize=(8, 4))
            plt.plot(sim_row.cpu().float().numpy())
            plt.title(f"Q[{pos}] · Q[j] (Head {h})")
            plt.xlabel("Token Position j")
            plt.ylabel("Dot Product")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"qq_dot_fixedpos{pos}_head{h}.png"))
            plt.close()

    # 3. Mean dot product across heads
    mean_dot = QQ_dot.mean(dim=0)  # [L, L]
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_dot.cpu().float(), cmap="coolwarm", center=0)
    plt.title("Mean QQ Dot Product across all heads")
    plt.xlabel("Token Position")
    plt.ylabel("Token Position")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qq_dot_mean_heatmap_all_heads.png"))
    plt.close()

    print(f"[Done] All dot product plots saved to {output_dir}")

analyze_QQ_dot_product(q_raw, selected_heads=[0,5,15,30])