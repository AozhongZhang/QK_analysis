import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
import json


import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def visualize_qk_attention_projection(K, Q_list, layer=30, head=0, topk_q_idx=[0,1,2,3]):
    """
    K: [L, d] tensor
    Q_list: List of Q tensors, e.g., Q[topk_idx] → [num_q, d]
    """
    # Step 1: PCA 
    pca = PCA(n_components=2)
    K_2d = pca.fit_transform(K)
    
    
    # Step 2: 可视化 K 和 Q
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
    # Step 3: 可视化 QK-attended vector
    # for i, hat_k in enumerate(hat_K_list):
    #     hat_proj = pca.transform(hat_k.cpu().numpy())
    #     plt.scatter(hat_proj[:, 0], hat_proj[:, 1], color='black', marker='x', s=40, label=f"Q_{i}@K")

    plt.title(f"K Projection + Q Projection (Head {head}, Layer{layer})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/azzhang/QK_analysis/qk_cone_64k/layer{layer}_{head}_qk_beforerope_64k.png")


def plot_concatall_qk_shared_pca_subplots(
    k_4k_before, q_4k_before,
    k_4k_after, q_4k_after,
    k_64k_before, q_64k_before,
    k_64k_after, q_64k_after,
    title_prefix="Q/K PCA View",
    layer=0, head=0
):
    """
    parameters:
    - k_xxx: Tensor, shape [seq_len, head_dim]
    - q_xxx: List[Tensor], 每个是 [seq_len, head_dim]
    """

    # Step 1: PCA colloct all vector to PCA
    all_vecs = torch.cat([
        k_4k_before, *q_4k_before,
        k_4k_after, *q_4k_after,
        k_64k_before, *q_64k_before,
        k_64k_after, *q_64k_after
    ], dim=0)  # [total_seq, dim]

    # Step 2: fit PCA
    pca = PCA(n_components=2)
    pca.fit(all_vecs.cpu().numpy())
    V_shared = torch.from_numpy(pca.components_.T).to(all_vecs.device)  # [dim, 2]

    # Step 3: prepaer data
    entries = [
        ("4K Before RoPE", k_4k_before, q_4k_before),
        ("4K After RoPE", k_4k_after, q_4k_after),
        ("64K Before RoPE", k_64k_before, q_64k_before),
        ("64K After RoPE", k_64k_after, q_64k_after),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    for i, (title, k, q_list) in enumerate(entries):
        ax = axs[i]
        k_proj = k @ V_shared  # [seq_len, 2]
        ax.scatter(k_proj[:, 0].cpu(), k_proj[:, 1].cpu(), label="K", c='black', s=10, alpha=0.7)

        for qi, q in enumerate(q_list):
            q_proj = q @ V_shared
            ax.scatter(q_proj[:, 0].cpu(), q_proj[:, 1].cpu(), label=f"Q{qi}", s=10, alpha=0.5)

        ax.set_title(f"{title_prefix} - {title}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # plt.savefig(f"/home/azzhang/QK_analysis/qk_cone_samcom/layer_qk_samecom.png")
    plt.savefig(f"/home/azzhang/QK_analysis/qk_cone_plot_sam_3_all/layer_{layer}_head_{head}_qk_samecom.png")
    # plt.show()


import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_qk_split_projection_from_4k_before(
    k_4k_before, q_4k_before,
    k_4k_after, q_4k_after,
    k_64k_before, q_64k_before,
    k_64k_after, q_64k_after,
    title_prefix="Q/K Separate PCA View (from 4K-before)",
    layer=0, head=0
):
    """
    Q and K are PCA-ed on 4K-before to get their respective projection bases.
    All Q are subsequently projected into Q-space and K into K-space, and are drawn in the same figure.
    One sub-figure for each group (4 in total).
    """

    # Step 1: get the PCA subspace of K 
    pca_k = PCA(n_components=2)
    pca_k.fit(k_4k_after.cpu().numpy())
    V_k = torch.from_numpy(pca_k.components_.T).to(k_4k_after.device)  # [dim, 2]

    # Step 2: get the PCA subspace of Q (concat 4 Q)
    q_cat = torch.cat(q_4k_after, dim=0)  # [4 * seq_len, dim]
    pca_q = PCA(n_components=2)
    pca_q.fit(q_cat.cpu().numpy())
    V_q = torch.from_numpy(pca_q.components_.T).to(q_cat.device)  # [dim, 2]

    # Step 3: prepare data
    entries = [
        ("4K Before RoPE", k_4k_before, q_4k_before),
        ("4K After RoPE", k_4k_after, q_4k_after),
        ("64K Before RoPE", k_64k_before, q_64k_before),
        ("64K After RoPE", k_64k_after, q_64k_after),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    for i, (title, k, q_list) in enumerate(entries):
        ax = axs[i]

        # project K into K subspace
        k_proj = k @ V_k  # [seq_len, 2]
        ax.scatter(k_proj[:, 0].cpu(), k_proj[:, 1].cpu(), label="K", c='black', s=10, alpha=0.7)

        # project each Q into Q subspace
        for qi, q in enumerate(q_list):
            q_proj = q @ V_q  # [seq_len, 2]
            ax.scatter(q_proj[:, 0].cpu(), q_proj[:, 1].cpu(), label=f"Q{qi}", s=10, alpha=0.5)

        ax.set_title(f"{title_prefix} - {title}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"/home/azzhang/QK_analysis/qk_cone_plot_sam_3.1/layer_{layer}_head_{head}_qk_samecom.png")
    # plt.show()


import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_qk_shared_projection_from_4k_before(
    k_4k_before, q_4k_before,
    k_4k_after, q_4k_after,
    k_64k_before, q_64k_before,
    k_64k_after, q_64k_after,
    title_prefix="Q/K PCA View (Fixed from 4K-before)",
    layer=0, head=0,
):
    """
    使用 Q/K 4K-before-RoPE 的投影主方向，对所有组投影到同一空间，生成 4 子图。
    所有输入：
        k: Tensor [seq_len, dim]
        q: List[Tensor [seq_len, dim]]
    """

    # Step 1: 拿 Q/K 4K-before 做 PCA
    ref_matrix = torch.cat([k_4k_before] + q_4k_before, dim=0)  # [total_tokens, dim]
    pca = PCA(n_components=2)
    pca.fit(ref_matrix.cpu().numpy())
    V_shared = torch.from_numpy(pca.components_.T).to(ref_matrix.device)  # [dim, 2]

    # Step 2: 定义四个数据块
    entries = [
        ("4K Before RoPE", k_4k_before, q_4k_before),
        ("4K After RoPE", k_4k_after, q_4k_after),
        ("64K Before RoPE", k_64k_before, q_64k_before),
        ("64K After RoPE", k_64k_after, q_64k_after),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    for i, (title, k, q_list) in enumerate(entries):
        ax = axs[i]

        # 投影 K
        k_proj = k @ V_shared
        ax.scatter(k_proj[:, 0].cpu(), k_proj[:, 1].cpu(), label="K", c='black', s=10, alpha=0.7)

        # 投影 Qs
        for qi, q in enumerate(q_list):
            q_proj = q @ V_shared
            ax.scatter(q_proj[:, 0].cpu(), q_proj[:, 1].cpu(), label=f"Q{qi}", s=10, alpha=0.5)

        ax.set_title(f"{title_prefix} - {title}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # plt.legend()
    # plt.grid(True)
    plt.savefig(f"/home/azzhang/QK_analysis/qk_cone_plot_sam_3_beforeqkall/layer_{layer}_head_{head}_qk_samecom.png.png")
    # plt.show()



def plot_joint_pca_k_q_list(K: torch.Tensor, Q_list: list, k: int = 2):
    """
    对 K 和多个 Q 一起做联合 PCA，并将它们投影到同一个空间绘图。

    参数:
    - K: Tensor, shape [seq_len, head_dim]
    - Q_list: List[Tensor], 每个元素 shape [seq_len, head_dim]
    - k: 主成分个数，推荐为2以便绘图
    """
    assert k == 2, "目前只支持2D可视化"

    # 拼接所有数据
    all_data = [K] + Q_list
    all_stacked = torch.cat(all_data, dim=0)  # [seq_len * (1 + len(Q_list)), head_dim]
    
    # 做 PCA
    mean = all_stacked.mean(dim=0, keepdim=True)
    centered = all_stacked - mean
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    P = Vh[:k, :]  # [k, head_dim]

    # 投影 K
    K_proj = (K - mean) @ P.T  # [seq_len, 2]

    # 投影每个 Q
    Q_proj_list = [(Q - mean) @ P.T for Q in Q_list]

    # 绘图
    plt.figure(figsize=(7, 6))
    plt.scatter(K_proj[:, 0], K_proj[:, 1], label='K', alpha=0.8, marker='o')

    for i, Q_proj in enumerate(Q_proj_list):
        plt.scatter(Q_proj[:, 0], Q_proj[:, 1], label=f'Q[{i}]', alpha=0.5, marker='x')

    plt.title("Joint PCA Projection of K and Q_list")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"/home/azzhang/QK_analysis/qk_cone_samcom/layer_qk_samecom.png")
    # plt.show()




# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
target_length_64k = "64k"
target_length_4k = "4k"

prompt_64k = prompts[target_length_64k]
prompt_4k = prompts[target_length_4k]
inputs_64k = tokenizer(prompt_64k, return_tensors="pt").to(model.device)
inputs_4k = tokenizer(prompt_4k, return_tensors="pt").to(model.device)
seq_len_64 = inputs_64k["input_ids"].shape[1]
seq_len_4 = inputs_4k["input_ids"].shape[1]
print(seq_len_64)
print(seq_len_4)

cache_4k = {}
cache_64k = {}
target_layer = 0
# target_head = 2

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
    if seqlen == seq_len_4:
        print(seq_len_4)
        q = q.view(bsz, seqlen, num_heads_q, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)
        # v = v.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)

        # Q、K before rope
        cache_4k["q_raw"] = q.detach().cpu()
        cache_4k["k_raw"] = k.detach().cpu()
        # cache["v"] = v.detach().cpu()

        # cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        cache_4k["q_rope"] = q_rope.detach().cpu()
        cache_4k["k_rope"] = k_rope.detach().cpu()
    else:
        print(seq_len_64)
        q = q.view(bsz, seqlen, num_heads_q, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)
        # v = v.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)

        # Q、K before rope
        cache_64k["q_raw"] = q.detach().cpu()
        cache_64k["k_raw"] = k.detach().cpu()
        # cache["v"] = v.detach().cpu()

        # cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        cache_64k["q_rope"] = q_rope.detach().cpu()
        cache_64k["k_rope"] = k_rope.detach().cpu()

    
    return self._orig_forward(hidden_states, position_embeddings, *args, **kwargs)

# 注入 patch
attn_layer = model.model.layers[target_layer].self_attn
attn_layer._orig_forward = attn_layer.forward
attn_layer.forward = MethodType(patched_forward, attn_layer)

with torch.no_grad():
    outputs = model(**inputs_4k)
with torch.no_grad():
    outputs = model(**inputs_64k)

Q_4k = cache_4k["q_raw"].squeeze(0)  # shape: (num_heads, seq_len, head_dim) for one layer (32, 4k, 1024//32)
K_4k = cache_4k["k_raw"].squeeze(0) # (8, 4k, 1024//8)

Q_64k = cache_64k["q_raw"].squeeze(0)  # shape: (num_heads, seq_len, head_dim)
K_64k = cache_64k["k_raw"].squeeze(0)

Q_4k_rope = cache_4k["q_rope"].squeeze(0)  # shape: (num_heads, seq_len, head_dim)
K_4k_rope = cache_4k["k_rope"].squeeze(0)

Q_64k_rope = cache_64k["q_rope"].squeeze(0)  # shape: (num_heads, seq_len, head_dim)
K_64k_rope = cache_64k["k_rope"].squeeze(0)
# V = cache["v"].squeeze(0)


# target_head = 2
for target_head in range(8):
    Q_head_0_4k = Q_4k[4*target_head].float()  # shape [L, d]
    Q_head_1_4k = Q_4k[4*target_head+1].float()
    Q_head_2_4k = Q_4k[4*target_head+2].float()
    Q_head_3_4k = Q_4k[4*target_head+3].float()
    K_head_4k = K_4k[target_head].float()
            # print("Q head shape: ", Q_head.shape)
    print("K_4k head shape: ", K_head_4k.shape)



    Q_list_4k = []
    Q_list_4k.append(Q_head_0_4k)
    Q_list_4k.append(Q_head_1_4k)
    Q_list_4k.append(Q_head_2_4k)
    Q_list_4k.append(Q_head_3_4k)

    # plot_joint_pca_k_q_list(K=K_head_4k, Q_list=Q_list_4k)

    # after rope 4k
    Q_head_0_4k_rope = Q_4k_rope[4*target_head].float()  # shape [L, d]
    Q_head_1_4k_rope = Q_4k_rope[4*target_head+1].float()
    Q_head_2_4k_rope = Q_4k_rope[4*target_head+2].float()
    Q_head_3_4k_rope = Q_4k_rope[4*target_head+3].float()
    K_head_4k_rope = K_4k_rope[target_head].float()
            # print("Q head shape: ", Q_head.shape)
    print("K_4k_rope head shape: ", K_head_4k.shape)

    Q_list_4k_rope = []
    Q_list_4k_rope.append(Q_head_0_4k_rope)
    Q_list_4k_rope.append(Q_head_1_4k_rope)
    Q_list_4k_rope.append(Q_head_2_4k_rope)
    Q_list_4k_rope.append(Q_head_3_4k_rope)

    # plot_joint_pca_k_q_list(K=K_head_4k_rope, Q_list=Q_list_4k_rope)

    # before rope 64k
    Q_head_0_64k = Q_64k[4*target_head].float()  # shape [L, d]
    Q_head_1_64k = Q_64k[4*target_head+1].float()
    Q_head_2_64k = Q_64k[4*target_head+2].float()
    Q_head_3_64k = Q_64k[4*target_head+3].float()
    K_head_64k = K_64k[target_head].float()
            # print("Q head shape: ", Q_head.shape)
    print("K_64k head shape: ", K_head_64k.shape)

    Q_list_64k = []
    Q_list_64k.append(Q_head_0_64k)
    Q_list_64k.append(Q_head_1_64k)
    Q_list_64k.append(Q_head_2_64k)
    Q_list_64k.append(Q_head_3_64k)

    # plot_joint_pca_k_q_list(K=K_head_64k, Q_list=Q_list_64k)

    # after rope 64k
    Q_head_0_64k_rope = Q_64k_rope[4*target_head].float()  # shape [L, d]
    Q_head_1_64k_rope = Q_64k_rope[4*target_head+1].float()
    Q_head_2_64k_rope = Q_64k_rope[4*target_head+2].float()
    Q_head_3_64k_rope = Q_64k_rope[4*target_head+3].float()
    K_head_64k_rope = K_64k_rope[target_head].float()
    print("K_64k_rope head shape: ", K_head_64k_rope.shape)

    Q_list_64k_rope = []
    Q_list_64k_rope.append(Q_head_0_64k_rope)
    Q_list_64k_rope.append(Q_head_1_64k_rope)
    Q_list_64k_rope.append(Q_head_2_64k_rope)
    Q_list_64k_rope.append(Q_head_3_64k_rope)
    # plot_joint_pca_k_q_list(K=K_64k_rope, Q_list=Q_list_64k_rope)

    plot_qk_shared_projection_from_4k_before(k_4k_before=K_head_4k, q_4k_before=Q_list_4k, 
                                            k_4k_after=K_head_4k_rope, q_4k_after=Q_list_4k_rope,
                                            k_64k_before=K_head_64k, q_64k_before=Q_list_64k,
                                            k_64k_after=K_head_64k_rope, q_64k_after=Q_list_64k_rope,
                                            layer=target_layer, head=target_head)



# visualize_qk_attention_projection(K_head, Q_list, target_layer, target_head)



