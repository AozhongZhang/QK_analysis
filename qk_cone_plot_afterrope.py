import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import MethodType
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import seaborn as sns
from sklearn.manifold import MDS
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import json

def visualize_qk_attention_projection(K, Q_list, layer=30, head=0, topk_q_idx=[0,1,2,3]):
    """
    K: [L, d] tensor
    Q_list: List of Q tensors, e.g., Q[topk_idx] → [num_q, d]
    """
    # Step 1: PCA 
    pca = PCA(n_components=2)
    K_2d = pca.fit_transform(K)
    
    
    # Step 2: vis K 和 Q
    plt.figure(figsize=(8, 6))
    plt.scatter(K_2d[:, 0], K_2d[:, 1], color='blue', s=5, label='K')

    colors = ['orange', 'green', 'purple', 'yellow']
    hat_K_list = []

    for i, q in enumerate(Q_list):
        # q = q.unsqueeze(0)  # [1, d]
        scores = torch.matmul(q, K.T) / (K.shape[1] ** 0.5)  # [1, L]
        attn = torch.softmax(scores, dim=-1)  # [1, L]
        # weighted_K = torch.matmul(attn, K)  # [1, d]
        hat_K_list.append(attn)

        q_proj = pca.transform(q.cpu().numpy())
        plt.scatter(q_proj[:, 0], q_proj[:, 1], label=f"Q_{i}", color=colors[i], s=10)
    # pca_qk = PCA(n_components=2)
    # print(hat_K_list[0].shape)
    # QK_0 = pca_qk.fit_transform(hat_K_list[0]) 

    # plt.scatter(QK_0[:, 0], QK_0[:, 1], color='black', marker='x', s=40, label=f"Q_{0}@K")
    # Step 3: vis QK-attended vector
    # for i, hat_k in enumerate(hat_K_list):
    #     hat_proj = pca.transform(hat_k.cpu().numpy())
    #     plt.scatter(hat_proj[:, 0], hat_proj[:, 1], color='black', marker='x', s=40, label=f"Q_{i}@K")

    plt.title(f"K Projection + Q Projection (Head {head}, Layer{layer})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/azzhang/QK_analysis/qk_cone_64k/layer{layer}_{head}_qk_afterrope_64k.png")

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
target_length = "64k"

prompt = prompts[target_length]
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
seq_len_1 = inputs["input_ids"].shape[1]
print(seq_len_1)

cache = {}
target_layer = 10

def patched_forward(self, hidden_states, position_embeddings=None, *args, **kwargs):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    bsz, seqlen, dim = q.shape
    head_dim = self.head_dim
    # num_heads = self.num_heads
    num_heads_q = self.config.num_attention_heads
    num_heads_kv = self.config.num_key_value_heads
    # print(num_heads_q)
    # print(num_heads_kv)

    q = q.view(bsz, seqlen, num_heads_q, head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)

    # Q、K before rope
    # cache["q_raw"] = q.detach().cpu()
    # cache["k_raw"] = k.detach().cpu()
    # cache["v"] = v.detach().cpu()

    # cos, sin = self.rotary_emb(hidden_states, position_ids)
    cos, sin = position_embeddings
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    cache["q_rope"] = q_rope.detach().cpu()
    cache["k_rope"] = k_rope.detach().cpu()

    
    return self._orig_forward(hidden_states, position_embeddings, *args, **kwargs)

# insert patch
attn_layer = model.model.layers[target_layer].self_attn
attn_layer._orig_forward = attn_layer.forward
attn_layer.forward = MethodType(patched_forward, attn_layer)

with torch.no_grad():
    outputs = model(**inputs)

Q = cache["q_rope"].squeeze(0)  # shape: (num_heads, seq_len, head_dim)
K = cache["k_rope"].squeeze(0)
# V = cache["v"].squeeze(0)



for target_head in range(8):
    Q_head_0 = Q[4*target_head].float()  # shape [L, d]
    Q_head_1 = Q[4*target_head+1].float()
    Q_head_2 = Q[4*target_head+2].float()
    Q_head_3 = Q[4*target_head+3].float()
    K_head = K[target_head].float()
    print("Q head shape: ", K_head.shape)

    Q_list = []
    Q_list.append(Q_head_0)
    Q_list.append(Q_head_1)
    Q_list.append(Q_head_2)
    Q_list.append(Q_head_3)

    visualize_qk_attention_projection(K_head, Q_list, target_layer, target_head)



