from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from types import MethodType
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
# from sample_test imporsample


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
for i in range(32):
    cache = {}

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

    q_rope = cache["q_rope"].squeeze(0)
    k_rope = cache["k_rope"].squeeze(0)


    num_heads, seq_len, head_dim = q_raw.shape
    print(seq_len)


    # 初始化存储结果
    qq_sim_raw, kk_sim_raw, qk_sim_raw = [], [], []
    qq_sim_rope, kk_sim_rope, qk_sim_rope = [], [], []
    for head in range(num_heads):
        q_raw_h = q_raw[head]
        k_raw_h = k_raw[head]
        q_rope_h = q_rope[head]
        k_rope_h = k_rope[head]

        # Q-Q similarity before rope
        qq_raw = F.cosine_similarity(q_raw_h.unsqueeze(1), q_raw_h.unsqueeze(0), dim=-1).mean().item()
        qq_rope = F.cosine_similarity(q_rope_h.unsqueeze(1), q_rope_h.unsqueeze(0), dim=-1).mean().item()
        qq_sim_raw.append(qq_raw)
        qq_sim_rope.append(qq_rope)

        # K-K similarity
        kk_raw = F.cosine_similarity(k_raw_h.unsqueeze(1), k_raw_h.unsqueeze(0), dim=-1).mean().item()
        kk_rope = F.cosine_similarity(k_rope_h.unsqueeze(1), k_rope_h.unsqueeze(0), dim=-1).mean().item()
        kk_sim_raw.append(kk_raw)
        kk_sim_rope.append(kk_rope)

        # Q-K similarity
        qk_raw = F.cosine_similarity(q_raw_h.unsqueeze(1), k_raw_h.unsqueeze(0), dim=-1).mean().item()
        qk_rope = F.cosine_similarity(q_rope_h.unsqueeze(1), k_rope_h.unsqueeze(0), dim=-1).mean().item()
        qk_sim_raw.append(qk_raw)
        qk_sim_rope.append(qk_rope)


    import matplotlib.pyplot as plt
    import numpy as np



    qq_raw_sims = np.array(qq_sim_raw)
    kk_raw_sims = np.array(kk_sim_raw)
    qk_raw_sims = np.array(qk_sim_raw)

    # RoPE 后
    qq_rope_sims = np.array(qq_sim_rope)
    kk_rope_sims = np.array(kk_sim_rope)
    qk_rope_sims = np.array(qk_sim_rope)

    num_heads = len(qq_raw_sims)
    heads = np.arange(num_heads)

    bar_width = 0.35
    offset = bar_width / 2

    # 创建图表
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

    # Q-Q 相似度
    axs[0].bar(heads - offset, qq_raw_sims, width=bar_width, label="Before RoPE")
    axs[0].bar(heads + offset, qq_rope_sims, width=bar_width, label="After RoPE")
    axs[0].set_title("Q-Q Cosine Similarity")
    axs[0].set_xticks(heads)
    axs[0].set_xlabel("Head")
    axs[0].grid(True, linestyle='--', alpha=0.3)

    # K-K 相似度
    axs[1].bar(heads - offset, kk_raw_sims, width=bar_width, label="Before RoPE")
    axs[1].bar(heads + offset, kk_rope_sims, width=bar_width, label="After RoPE")
    axs[1].set_title("K-K Cosine Similarity")
    axs[1].set_xticks(heads)
    axs[1].set_xlabel("Head")
    axs[1].grid(True, linestyle='--', alpha=0.3)

    # Q-K 相似度
    axs[2].bar(heads - offset, qk_raw_sims, width=bar_width, label="Before RoPE")
    axs[2].bar(heads + offset, qk_rope_sims, width=bar_width, label="After RoPE")
    axs[2].set_title("Q-K Cosine Similarity")
    axs[2].set_xticks(heads)
    axs[2].set_xlabel("Head")
    axs[2].grid(True, linestyle='--', alpha=0.3)

    # 图例与布局
    axs[0].set_ylabel("Average Cosine Similarity")
    axs[0].legend(loc="upper right")
    plt.suptitle("Comparison of Cosine Similarities Before and After RoPE")
    plt.tight_layout()




    # # 设置柱状图参数
    # bar_width = 0.25
    # plt.figure(figsize=(10, 5))

    # # 三组数据分别偏移绘制
    # plt.bar(heads - bar_width, qq_raw_sims, width=bar_width, label='Q-Q similarity')
    # plt.bar(heads, kk_raw_sims, width=bar_width, label='K-K similarity')
    # plt.bar(heads + bar_width, qk_raw_sims, width=bar_width, label='Q-K similarity')

    # # 图表美化
    # plt.xlabel("Head Index")
    # plt.ylabel("Average Cosine Similarity")
    # plt.title("Per-Head Q-Q, K-K, and Q-K Cosine Similarities for layer 3")
    # plt.xticks(heads)
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.3)
    # plt.tight_layout()
    plt.savefig(f"/home/azzhang/streaming-llm/cos_sim/layer_{i}_raw_rope.png")

# 显示图像
# plt.show()
