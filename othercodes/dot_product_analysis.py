from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from types import MethodType
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
# from sample_test imporsample
# import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# model_name = 'meta-llama/Llama-2-7b-hf'
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(A)
# 构造输入
# # prompt = "The quick brown fox jumps over the lazy dog."
# prompt = "Hello! I've been well. I hope you're doing well too."
# # length = 287
# # prompt = "As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war. While at times this works to their advantage, such as a successful incursion into Imperial territory, other orders cause certain members of the 422nd great distress. One such member, Gusurg, becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven, attached to the ideal of Darcsen independence proposed by their leader, Dahau. At the same time, elements within Gallian Army Command move to erase the Nameless in order to protect their own interests. Hounded by both allies and enemies, and combined with the presence of a traitor within their ranks, the 422nd desperately move to keep themselves alive while at the same time fight to help the Gallian war effort. This continues until the Nameless's commanding officer, Ramsey Crowe, who had been kept under house arrest, is escorted to the capital city of Randgriz in order to present evidence exonerating the weary soldiers and expose the real traitor, the Gallian General that had accused Kurt of Treason."
# # length 263+287=549
# # prompt = "As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war. While at times this works to their advantage, such as a successful incursion into Imperial territory, other orders cause certain members of the 422nd great distress. One such member, Gusurg, becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven, attached to the ideal of Darcsen independence proposed by their leader, Dahau. At the same time, elements within Gallian Army Command move to erase the Nameless in order to protect their own interests. Hounded by both allies and enemies, and combined with the presence of a traitor within their ranks, the 422nd desperately move to keep themselves alive while at the same time fight to help the Gallian war effort. This continues until the Nameless's commanding officer, Ramsey Crowe, who had been kept under house arrest, is escorted to the capital city of Randgriz in order to present evidence exonerating the weary soldiers and expose the real traitor, the Gallian General that had accused Kurt of Treason. Partly due to these events, and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire, the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force. This is short @-@ lived, however, as following Maximilian's defeat, Dahau and Calamity Raven move to activate an ancient Valkyrian super weapon within the Empire, kept secret by their benefactor. Without the support of Maximilian or the chance to prove themselves in the war with Gallia, it is Dahau's last trump card in creating a new Darcsen nation. As an armed Gallian force invading the Empire just following the two nations' cease @-@ fire would certainly wreck their newfound peace, Kurt decides to once again make his squad the Nameless, asking Crowe to list himself and all under his command as killed @-@ in @-@ action. Now owing allegiance to none other than themselves, the 422nd confronts Dahau and destroys the Valkyrian weapon. Each member then goes their separate ways in order to begin their lives anew."
# # length 884
# # prompt = "As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war. While at times this works to their advantage, such as a successful incursion into Imperial territory, other orders cause certain members of the 422nd great distress. One such member, Gusurg, becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven, attached to the ideal of Darcsen independence proposed by their leader, Dahau. At the same time, elements within Gallian Army Command move to erase the Nameless in order to protect their own interests. Hounded by both allies and enemies, and combined with the presence of a traitor within their ranks, the 422nd desperately move to keep themselves alive while at the same time fight to help the Gallian war effort. This continues until the Nameless's commanding officer, Ramsey Crowe, who had been kept under house arrest, is escorted to the capital city of Randgriz in order to present evidence exonerating the weary soldiers and expose the real traitor, the Gallian General that had accused Kurt of Treason. Partly due to these events, and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire, the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force. This is short @-@ lived, however, as following Maximilian's defeat, Dahau and Calamity Raven move to activate an ancient Valkyrian super weapon within the Empire, kept secret by their benefactor. Without the support of Maximilian or the chance to prove themselves in the war with Gallia, it is Dahau's last trump card in creating a new Darcsen nation. As an armed Gallian force invading the Empire just following the two nations' cease @-@ fire would certainly wreck their newfound peace, Kurt decides to once again make his squad the Nameless, asking Crowe to list himself and all under his command as killed @-@ in @-@ action. Now owing allegiance to none other than themselves, the 422nd confronts Dahau and destroys the Valkyrian weapon. Each member then goes their separate ways in order to begin their lives anew. The majority of material created for previous games, such as the BLiTZ system and the design of maps, was carried over. Alongside this, improvements were made to the game's graphics and some elements were expanded, such as map layouts, mission structure, and the number of playable units per mission. A part of this upgrade involved creating unique polygon models for each character's body. In order to achieve this, the cooperative elements incorporated into the second game were removed, as they took up a large portion of memory space needed for the improvements. They also adjusted the difficulty settings and ease of play so they could appeal to new players while retaining the essential components of the series' gameplay. The newer systems were decided upon early in development. The character designs were done by Raita Honjou, who had worked on the previous Valkyria Chronicles games. When creating the Nameless Squad, Honjou was faced with the same problem he had had during the first game: the military uniforms essentially destroyed character individuality, despite him needing to create unique characters the player could identify while maintaining a sense of reality within the Valkyria Chronicles world. The main color of the Nameless was black. As with the previous Valkyria games, Valkyria Chronicles III used the CANVAS graphics engine. The anime opening was produced by Production I.G."
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# seq_len = inputs["input_ids"].shape[1]
# print(seq_len_1)
with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
prompt_1k = prompts["1k"]
inputs = tokenizer(prompt_1k, return_tensors="pt").to(model.device)
seq_len_1 = inputs["input_ids"].shape[1]
print(seq_len_1)

# target_layer = 4
# inputs = data_sample(256, model.device)
Q_list = []
K_list = []
V_list = []
for i in range(32):
    cache = {}
    # patch forward
    # def patched_forward(self, hidden_states, attention_mask=None, position_ids=None,
    #                     past_key_value=None, output_attentions=False, use_cache=False,
    #                     **kwargs):
    def patched_forward(self, hidden_states, *args, **kwargs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        bsz, seqlen, dim = q.shape
        head_dim = self.head_dim
        # num_heads = self.num_heads
        num_heads_q = self.config.num_attention_heads
        num_heads_kv = self.config.num_key_value_heads
        # num_heads = 8

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

        # 继续调用原forward方法
        # return self._orig_forward(hidden_states, attention_mask, position_ids,
        #                         past_key_value, output_attentions, use_cache,
        #                         **kwargs)
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

    num_heads, seq_len, head_dim = q_raw.shape
    # print(seq_len)

    Q_list.append(q_raw)
    K_list.append(k_raw)
    V_list.append(v)

# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# 计算每层排序后的 self-similarity 曲线
# layer_curves = []
# for Q_layer in Q_list:
#     head_similarities = []
#     for h in range(Q_layer.shape[0]):
#         q_h = Q_layer[h]  # [seq_len, head_dim]
#         sim_matrix = F.cosine_similarity(q_h.unsqueeze(1), q_h.unsqueeze(0), dim=-1)
#         avg_sim = sim_matrix.mean().item()
#         head_similarities.append(avg_sim)
#     head_similarities_sorted = sorted(head_similarities)
#     layer_curves.append(head_similarities_sorted)

# # 转为矩阵 [num_layers, num_heads]
# sim_matrix_all_layers = torch.tensor(layer_curves)




def compute_similarity_matrix(A_list, B_list=None):
    all_layers = []
    for l in range(len(A_list)):
        sims = []
        for h in range(A_list[l].shape[0]):
            a = A_list[l][h]  # [seq_len, head_dim]
            b = B_list[l][h] if B_list is not None else a
            sim = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1).mean().item()
            sims.append(sim)
        sims_sorted = sorted(sims)
        all_layers.append(sims_sorted)
    return torch.tensor(all_layers)

def compute_dot_product_matrix(A_list, B_list=None):
    all_layers = []
    for l in range(len(A_list)):
        sims = []
        for h in range(A_list[l].shape[0]):
            a = A_list[l][h]  # [seq_len, head_dim]
            b = B_list[l][h] if B_list is not None else a  # [seq_len, head_dim]
            
            # dot product similarity matrix: [seq_len, seq_len]
            sim_matrix = torch.matmul(a, b.T)
            sim_mean = sim_matrix.mean().item()
            sims.append(sim_mean)
        
        sims_sorted = sorted(sims)
        all_layers.append(sims_sorted)
    return torch.tensor(all_layers)

def compute_grouped_dot_product(Q_list, K_list):
    """
    每个 Q[i] 对应 QK-grouped 的共享 K[j]，计算 dot product。
    适用于 LLaMA 中的 GQA 架构。
    返回 [num_layers, num_q_heads] 的相似度 tensor。
    """
    all_layers = []

    for l in range(len(Q_list)):
        Q = Q_list[l]  # [num_q, seq_len, head_dim]
        K = K_list[l]  # [num_k, seq_len, head_dim]
        num_q = Q.shape[0]
        num_k = K.shape[0]
        group_size = num_q // num_k

        sim_layer = []

        for q_id in range(num_q):
            k_id = q_id // group_size
            q = Q[q_id]  # [seq_len, head_dim]
            k = K[k_id]  # [seq_len, head_dim]

            sim = torch.matmul(q, k.T).mean().item()  # 平均 dot product
            sim_layer.append(sim)
        sim_layer = sorted(sim_layer)
        all_layers.append(sim_layer)

    return torch.tensor(all_layers)  # shape: [num_layers, num_q]


# 三种相似度
# QQ = compute_similarity_matrix(Q_list)
# KK = compute_similarity_matrix(K_list)
# QK = compute_similarity_matrix(Q_list, K_list)
QQ = compute_dot_product_matrix(Q_list)
KK = compute_dot_product_matrix(K_list)
QK = compute_grouped_dot_product(Q_list, K_list)

# 统一颜色范围
vmin = min(QQ.min(), KK.min(), QK.min()).item()
vmax = max(QQ.max(), KK.max(), QK.max()).item()

# 可视化
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for ax, data, title in zip(axs, [QQ, KK, QK], ["QQ Self-Similarity", "KK Self-Similarity", "QK Similarity"]):
    sns.heatmap(data, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel("Sorted Head Index")
    ax.set_ylabel("Layer Index")

plt.tight_layout()

plt.savefig(f"/home/azzhang/streaming-llm/dot_product/layer_llama3.png")
# num_layers = 32

# # 1️⃣ depth vs. input length: 每层平均所有 head 在每个 token 上的 self-similarity
# depth_vs_len = torch.zeros((num_layers, seq_len))
# for l in range(num_layers):
#     for t in range(seq_len):
#         sim_sum = 0
#         for h in range(num_heads):
#             q = Q_list[l][h]  # [seq_len, head_dim]
#             sim = F.cosine_similarity(q[t].unsqueeze(0), q, dim=-1).mean()
#             sim_sum += sim
#         depth_vs_len[l, t] = sim_sum / num_heads

# # 2️⃣ head vs. input length: 每个 head 在不同 token 上的相似度（跨层平均）
# head_vs_len = torch.zeros((num_heads, seq_len))
# for h in range(num_heads):
#     for t in range(seq_len):
#         sim_sum = 0
#         for l in range(num_layers):
#             q = Q_list[l][h]  # [seq_len, head_dim]
#             sim = F.cosine_similarity(q[t].unsqueeze(0), q, dim=-1).mean()
#             sim_sum += sim
#         head_vs_len[h, t] = sim_sum / num_layers

# # 对 head 按相似度均值排序
# sorted_idx = torch.argsort(head_vs_len.mean(dim=1))
# head_vs_len_sorted = head_vs_len[sorted_idx]

# # 画图
# plt.figure(figsize=(14, 5))

# plt.subplot(1, 2, 1)
# sns.heatmap(depth_vs_len, cmap="RdBu_r")
# plt.title("Depth vs Input Length (Avg over Heads)")
# plt.xlabel("Token Index")
# plt.ylabel("Layer Index")

# plt.subplot(1, 2, 2)
# sns.heatmap(head_vs_len_sorted, cmap="RdBu_r")
# plt.title("Sorted Head vs Input Length (Avg over Layers)")
# plt.xlabel("Token Index")
# plt.ylabel("Sorted Head Index")

# plt.tight_layout()






# # 画热力图
# plt.figure(figsize=(10, 6))
# sns.heatmap(sim_matrix_all_layers, cmap="RdBu_r", cbar=True, xticklabels=False, yticklabels=True)
# plt.title("Sorted Head Self-Similarity (Cosine) Across Layers")
# plt.xlabel("Sorted Head Index (within each layer)")
# plt.ylabel("Layer Index")
# plt.tight_layout()

# 你的真实 Q_tensor，shape [num_layers, num_heads, seq_len, head_dim]
# num_layers, num_heads, seq_len, head_dim = 12, 8, 128, 64
# Q_tensor = torch.randn(num_layers, num_heads, seq_len, head_dim)

# # 图 1：每一层在每个位置的平均 head 自相似度
# layer_input_sim = np.zeros((32, seq_len))
# for l in range(32):
#     for t in range(seq_len):
#         vecs = Q_list[l][:, t, :]  # [num_heads, head_dim]
#         sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)  # [H, H]
#         layer_input_sim[l, t] = sim.mean().item()

# # 图 2：按 head 排序，每个 head 在不同位置的响应（跨层平均）
# head_sim = []
# for h in range(32):
#     sims = []
#     for l in range(32):
#         x = Q_list[l][h]  # [seq_len, head_dim]
#         sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1).mean().item()
#         sims.append(sim)
#     head_sim.append(np.mean(sims))
# sorted_idx = np.argsort(head_sim)

# head_input_sim = np.zeros((32, seq_len))
# for i, h in enumerate(sorted_idx):
#     for t in range(seq_len):
#         # vecs = Q_list[:, h, t, :]  # [num_layers, head_dim]
#         vecs = torch.stack([Q_list[l][h, t, :] for l in range(len(Q_list))])  # shape: [num_layers, head_dim]
#         sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)  # [L, L]
#         head_input_sim[i, t] = sim.mean().item()

# # 绘图
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# sns.heatmap(layer_input_sim, ax=axes[0], cmap="YlOrRd", xticklabels=16,
#             yticklabels=[f"L{i}" for i in range(32)], cbar=True)
# axes[0].set_title("Layer × Input Position (avg over heads)")
# axes[0].set_xlabel("Input Position")
# axes[0].set_ylabel("Layer Index")

# sns.heatmap(head_input_sim, ax=axes[1], cmap="YlOrRd", xticklabels=16,
#             yticklabels=[f"H{i}" for i in sorted_idx], cbar=True)
# axes[1].set_title("Sorted Head × Input Position (avg over layers)")
# axes[1].set_xlabel("Input Position")
# axes[1].set_ylabel("Sorted Head Index")

# plt.tight_layout()


# def compute_self_similarity(X):
#     sims = []
#     for h in range(X.shape[0]):
#         x = X[h]  # [seq_len, head_dim]
#         sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
#         sims.append(sim.mean().item())
#     return sims

# def compute_qk_similarity(Q, K):
#     sims = []
#     for h in range(Q.shape[0]):
#         q = Q[h]
#         k = K[h]
#         sim = F.cosine_similarity(q.unsqueeze(1), k.unsqueeze(0), dim=-1)
#         sims.append(sim.mean().item())
#     return sims

# # 替换为你真实的 Q_list, K_list
# # num_layers, num_heads, seq_len, head_dim = 12, 8, 128, 64
# # Q_list = [torch.randn(num_heads, seq_len, head_dim) for _ in range(num_layers)]
# # K_list = [torch.randn(num_heads, seq_len, head_dim) for _ in range(num_layers)]

# # 相似度矩阵
# qq_sim = np.array([compute_self_similarity(Q) for Q in Q_list])
# kk_sim = np.array([compute_self_similarity(K) for K in K_list])
# qk_sim = np.array([compute_qk_similarity(Q_list[i], K_list[i]) for i in range(32)])

# # 统一颜色范围
# vmin = min(qq_sim.min(), kk_sim.min(), qk_sim.min())
# vmax = max(qq_sim.max(), kk_sim.max(), qk_sim.max())

# # 绘制图像
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# cmap = "RdBu_r"  # 红白蓝渐变，红色为高

# heatmap_kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap,
#                       xticklabels=[f"H{i}" for i in range(num_heads)],
#                       cbar=True, annot=False)

# sns.heatmap(qq_sim, ax=axes[0], yticklabels=[f"L{i}" for i in range(32)], **heatmap_kwargs)
# axes[0].set_title("QQ Similarity")

# sns.heatmap(kk_sim, ax=axes[1], yticklabels=False, **heatmap_kwargs)
# axes[1].set_title("KK Similarity")

# sns.heatmap(qk_sim, ax=axes[2], yticklabels=False, **heatmap_kwargs)
# axes[2].set_title("QK Similarity")

# plt.suptitle("Cosine Similarity of Attention Heads Across Layers", fontsize=16)
# plt.tight_layout()
# plt.show()

# -----------  计算每个 head 的自相似平均值 -----------
# def compute_head_self_sim(Q):
#     """Q: [num_heads, seq_len, head_dim]"""
#     sims = []
#     for h in range(Q.shape[0]):
#         q_h = Q[h]  # [seq_len, head_dim]
#         sim_matrix = F.cosine_similarity(q_h.unsqueeze(1), q_h.unsqueeze(0), dim=-1)  # [seq_len, seq_len]
#         sims.append(sim_matrix.mean().item())
#     return sims



# # 所有层的 head 自相似度，shape: [num_layers, num_heads]
# similarity_matrix = np.array([compute_head_self_sim(Q) for Q in K_list])

# # 画热力图
# plt.figure(figsize=(10, 6))
# sns.heatmap(similarity_matrix, cmap="YlGnBu", annot=False, cbar=True,
#             xticklabels=[f"H{i}" for i in range(32)],
#             yticklabels=[f"L{i}" for i in range(32)])
# plt.xlabel("Head Index")
# plt.ylabel("Layer Index")
# plt.title("Head Self-Similarity Across Layers")
# plt.tight_layout()






