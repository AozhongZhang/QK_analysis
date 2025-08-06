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

model_name = 'meta-llama/Llama-2-7b-hf'
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
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
i = 0
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

def plot_pca_heads(matrix: torch.Tensor, target_layer):
    # 假设 all_q: [num_heads, seq_len, head_dim]
    num_heads, seq_len, head_dim = matrix.shape
    print(num_heads)
    print(seq_len)
    print(head_dim)

    # reshape 为 [num_heads * seq_len, head_dim]
    matrix_reshaped = matrix.reshape(num_heads * seq_len, head_dim).cpu().numpy()

    # 为每个向量记录它来自哪个 head
    head_ids = np.repeat(np.arange(num_heads), seq_len)

    pca = PCA(n_components=2)
    q_pca = pca.fit_transform(matrix_reshaped)  # shape: [num_heads * seq_len, 2]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 使用离散 colormap
    cmap = cm.get_cmap("tab20", num_heads)
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
    ax.set_title("PCA of K vectors across all heads")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()      
    plt.savefig(f"/home/azzhang/streaming-llm/head_cluster_llama3/layer{target_layer}_v_raw.png")

plot_pca_heads(v, i)

