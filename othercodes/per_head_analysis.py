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

model_name = 'meta-llama/Llama-2-7b-hf'
# model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(A)
# 构造输入
# prompt = "The quick brown fox jumps over the lazy dog."
# prompt = "Hello! I've been well. I hope you're doing well too."
# length = 287
prompt = "As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war. While at times this works to their advantage, such as a successful incursion into Imperial territory, other orders cause certain members of the 422nd great distress. One such member, Gusurg, becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven, attached to the ideal of Darcsen independence proposed by their leader, Dahau. At the same time, elements within Gallian Army Command move to erase the Nameless in order to protect their own interests. Hounded by both allies and enemies, and combined with the presence of a traitor within their ranks, the 422nd desperately move to keep themselves alive while at the same time fight to help the Gallian war effort. This continues until the Nameless's commanding officer, Ramsey Crowe, who had been kept under house arrest, is escorted to the capital city of Randgriz in order to present evidence exonerating the weary soldiers and expose the real traitor, the Gallian General that had accused Kurt of Treason."
# length 263+287=549
# prompt = "As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war. While at times this works to their advantage, such as a successful incursion into Imperial territory, other orders cause certain members of the 422nd great distress. One such member, Gusurg, becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven, attached to the ideal of Darcsen independence proposed by their leader, Dahau. At the same time, elements within Gallian Army Command move to erase the Nameless in order to protect their own interests. Hounded by both allies and enemies, and combined with the presence of a traitor within their ranks, the 422nd desperately move to keep themselves alive while at the same time fight to help the Gallian war effort. This continues until the Nameless's commanding officer, Ramsey Crowe, who had been kept under house arrest, is escorted to the capital city of Randgriz in order to present evidence exonerating the weary soldiers and expose the real traitor, the Gallian General that had accused Kurt of Treason. Partly due to these events, and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire, the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force. This is short @-@ lived, however, as following Maximilian's defeat, Dahau and Calamity Raven move to activate an ancient Valkyrian super weapon within the Empire, kept secret by their benefactor. Without the support of Maximilian or the chance to prove themselves in the war with Gallia, it is Dahau's last trump card in creating a new Darcsen nation. As an armed Gallian force invading the Empire just following the two nations' cease @-@ fire would certainly wreck their newfound peace, Kurt decides to once again make his squad the Nameless, asking Crowe to list himself and all under his command as killed @-@ in @-@ action. Now owing allegiance to none other than themselves, the 422nd confronts Dahau and destroys the Valkyrian weapon. Each member then goes their separate ways in order to begin their lives anew."
# length 884
# prompt = "As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war. While at times this works to their advantage, such as a successful incursion into Imperial territory, other orders cause certain members of the 422nd great distress. One such member, Gusurg, becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven, attached to the ideal of Darcsen independence proposed by their leader, Dahau. At the same time, elements within Gallian Army Command move to erase the Nameless in order to protect their own interests. Hounded by both allies and enemies, and combined with the presence of a traitor within their ranks, the 422nd desperately move to keep themselves alive while at the same time fight to help the Gallian war effort. This continues until the Nameless's commanding officer, Ramsey Crowe, who had been kept under house arrest, is escorted to the capital city of Randgriz in order to present evidence exonerating the weary soldiers and expose the real traitor, the Gallian General that had accused Kurt of Treason. Partly due to these events, and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire, the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force. This is short @-@ lived, however, as following Maximilian's defeat, Dahau and Calamity Raven move to activate an ancient Valkyrian super weapon within the Empire, kept secret by their benefactor. Without the support of Maximilian or the chance to prove themselves in the war with Gallia, it is Dahau's last trump card in creating a new Darcsen nation. As an armed Gallian force invading the Empire just following the two nations' cease @-@ fire would certainly wreck their newfound peace, Kurt decides to once again make his squad the Nameless, asking Crowe to list himself and all under his command as killed @-@ in @-@ action. Now owing allegiance to none other than themselves, the 422nd confronts Dahau and destroys the Valkyrian weapon. Each member then goes their separate ways in order to begin their lives anew. The majority of material created for previous games, such as the BLiTZ system and the design of maps, was carried over. Alongside this, improvements were made to the game's graphics and some elements were expanded, such as map layouts, mission structure, and the number of playable units per mission. A part of this upgrade involved creating unique polygon models for each character's body. In order to achieve this, the cooperative elements incorporated into the second game were removed, as they took up a large portion of memory space needed for the improvements. They also adjusted the difficulty settings and ease of play so they could appeal to new players while retaining the essential components of the series' gameplay. The newer systems were decided upon early in development. The character designs were done by Raita Honjou, who had worked on the previous Valkyria Chronicles games. When creating the Nameless Squad, Honjou was faced with the same problem he had had during the first game: the military uniforms essentially destroyed character individuality, despite him needing to create unique characters the player could identify while maintaining a sense of reality within the Valkyria Chronicles world. The main color of the Nameless was black. As with the previous Valkyria games, Valkyria Chronicles III used the CANVAS graphics engine. The anime opening was produced by Production I.G."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
seq_len_1 = inputs["input_ids"].shape[1]
print(seq_len_1)

# target_layer = 4
# inputs = data_sample(256, model.device)
# for i in range(5,32):
cache = {}
i = 4
# patch forward
def patched_forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        **kwargs):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    bsz, seqlen, dim = q.shape
    head_dim = self.head_dim
    num_heads = self.num_heads

    q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)

    # 存储RoPE之前的Q、K
    cache["q_raw"] = q.detach().cpu()
    cache["k_raw"] = k.detach().cpu()
    cache["v"] = v.detach().cpu()

    cos, sin = self.rotary_emb(v, position_ids)  # 注意这里是 value_states 用于生成位置编码
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    cache["q_rope"] = q_rope.detach().cpu()
    cache["k_rope"] = k_rope.detach().cpu()

    # 继续调用原forward方法
    return self._orig_forward(hidden_states, attention_mask, position_ids,
                                past_key_value, output_attentions, use_cache,
                                **kwargs)

# 注入 patch
attn_layer = model.model.layers[i].self_attn
attn_layer._orig_forward = attn_layer.forward
attn_layer.forward = MethodType(patched_forward, attn_layer)

with torch.no_grad():
    outputs = model(**inputs)



q_raw = cache["q_raw"].squeeze(0)  # shape: (num_heads, seq_len, head_dim)
k_raw = cache["k_raw"].squeeze(0)
v = cache["v"].squeeze(0)

q_rope = cache["q_rope"].squeeze(0)
k_rope = cache["k_rope"].squeeze(0)

# k_raw_head = k_raw[0]
K_raw = k_raw.permute(1, 0, 2).contiguous() 
K_raw= K_raw.view(K_raw.shape[0], -1)
V_raw = v.permute(1, 0, 2).contiguous() 
V_raw= V_raw.view(V_raw.shape[0], -1)

K_raw = np.abs(K_raw.cpu().numpy())
V_raw = np.abs(V_raw.cpu().numpy())
seq_len, dim = K_raw.shape
X, Y = np.meshgrid(np.arange(dim), np.arange(seq_len))  # [token, column]
Z_k = K_raw
Z_v = V_raw

fig = plt.figure(figsize=(12, 5))

# 🔹 左图：Key
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Y, X, Z_k, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9)
ax1.set_title("Key", fontsize=12)
ax1.set_xlabel("Token")
ax1.set_ylabel("Column")
ax1.set_zlabel("Value")

# 可选添加高亮线（如顶部一行）
ax1.plot([seq_len - 1] * dim, np.arange(dim), K_raw[-1, :], color='green', linewidth=1.5)

# 🔹 右图：Value
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Y, X, Z_v, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9)
ax2.set_title("Value", fontsize=12)
ax2.set_xlabel("Token")
ax2.set_ylabel("Column")
ax2.set_zlabel("Value")

# 可选添加对角线
k = min(seq_len, dim)
diag_x = np.arange(k)
diag_y = np.arange(k)
diag_z = Z_v[diag_x, diag_y]
ax2.plot(diag_x, diag_y, diag_z, color='yellow', linewidth=2)

plt.tight_layout()
        
plt.savefig(f"/home/azzhang/streaming-llm/per_head_value/layer{i}_raw.png")








# for head in range(32):
#     # 假设 head_matrix 是 numpy 数组 [seq_len, dim]
#     k_raw_head = k_raw[head]
#     matrix_k = np.abs(k_raw_head.cpu().numpy())  # shape [seq_len, head_dim]
#     matrix_v = np.abs(v[head].cpu().numpy())

#     seq_len, dim = matrix_k.shape
#     X, Y = np.meshgrid(np.arange(dim), np.arange(seq_len))  # [token, column]
#     Z_k = matrix_k
#     Z_v = matrix_v

#     fig = plt.figure(figsize=(12, 5))

#     # 🔹 左图：Key
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.plot_surface(Y, X, Z_k, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9)
#     ax1.set_title("Key Head", fontsize=12)
#     ax1.set_xlabel("Token")
#     ax1.set_ylabel("Column")
#     ax1.set_zlabel("Value")

#     # 可选添加高亮线（如顶部一行）
#     ax1.plot([seq_len - 1] * dim, np.arange(dim), matrix_k[-1, :], color='green', linewidth=1.5)

#     # 🔹 右图：Value
#     ax2 = fig.add_subplot(122, projection='3d')
#     ax2.plot_surface(Y, X, Z_v, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9)
#     ax2.set_title("Value Head", fontsize=12)
#     ax2.set_xlabel("Token")
#     ax2.set_ylabel("Column")
#     ax2.set_zlabel("Value")

#     # 可选添加对角线
#     k = min(seq_len, dim)
#     diag_x = np.arange(k)
#     diag_y = np.arange(k)
#     diag_z = Z_v[diag_x, diag_y]
#     ax2.plot(diag_x, diag_y, diag_z, color='yellow', linewidth=2)

#     plt.tight_layout()

#     # fig = plt.figure(figsize=(8, 5))
#     # ax = fig.add_subplot(111, projection='3d')

#     # # 主图
#     # ax.plot_surface(Y, X, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9)

#     # # 添加一条对角线（从 token=0,column=0 到 token=min(seq_len,dim)-1）
#     # k = min(seq_len, dim)
#     # diag_x = np.arange(k)
#     # diag_y = np.arange(k)
#     # diag_z = Z[diag_x, diag_y]
#     # ax.plot(diag_x, diag_y, diag_z, color='yellow', linewidth=2)

#     # # 标签
#     # ax.set_xlabel("Token Index")
#     # ax.set_ylabel("Column Index")
#     # ax.set_zlabel("Value")
#     # ax.set_title(f"3D Visualization of Head 0")
#     # plt.tight_layout()
#     # plt.show()


        
#     plt.savefig(f"/home/azzhang/streaming-llm/per_head_value/layer{i}_head{head}_raw.png")

