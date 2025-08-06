from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from types import MethodType
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
# from sample_test imporsample
# import torch
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
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

# q_rope = cache["q_rope"].squeeze(0)
# k_rope = cache["k_rope"].squeeze(0)

def plot_pca_heads(matrix: torch.Tensor, target_layer, length, target_matrix):
    # 假设 all_q: [num_heads, seq_len, head_dim]
    num_heads, seq_len, head_dim = matrix.shape
    print(num_heads)
    print(seq_len)
    print(head_dim)

    # reshape 为 [num_heads * seq_len, head_dim]
    matrix_reshaped = matrix.reshape(num_heads * seq_len, head_dim).float().cpu().numpy()

    # 为每个向量记录它来自哪个 head
    head_ids = np.repeat(np.arange(num_heads), seq_len)

    pca = PCA(n_components=2)
    q_pca = pca.fit_transform(matrix_reshaped)  # shape: [num_heads * seq_len, 2]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 使用离散 colormap
    # cmap = cm.get_cmap("tab20", num_heads)
    cmap = plt.get_cmap("tab20", num_heads)
    scatter = ax.scatter(
        q_pca[:, 0], q_pca[:, 1],
        c=head_ids,
        cmap=cmap,
        s=10,
        alpha=0.7
    )

    # 添加 colorbar（用于标注 Head ID）
    cbar = plt.colorbar(scatter, ticks=np.arange(num_heads))
    cbar.set_label("Head ID")

    # 图示设置
    ax.set_title(f"PCA of {target_matrix} vectors across all heads")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()      
    plt.savefig(f"/home/azzhang/streaming-llm/cluster_llama3/{length}/layer{target_layer}_{target_matrix}_raw.png")

plot_pca_heads(q_raw, i, target_length, "Q")
plot_pca_heads(k_raw, i, target_length, "K")
plot_pca_heads(v, i, target_length, "V")

