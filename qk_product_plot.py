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


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
seq_lengths = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
# Q_list = []
# K_list = []

collected_k_outputs = []
collected_q_outputs = []
def k_proj_hook(module, input, output):
    """
    module: The layer that produced this output (k_proj).
    input:  The input to k_proj.
    output: The output from k_proj (shape [batch_size, seq_len, hidden_dim]).
    """
    # Detach to avoid growing the autograd graph
    bsz, seq_len, hidden = output.shape
    output = output.view(bsz, seq_len, 8, (hidden//8)).transpose(1, 2).squeeze(0)
    collected_k_outputs.append(output.detach().cpu())

def q_proj_hook(module, input, output):
    """
    Same logic as k_proj_hook, but for q_proj.
    """
    bsz, seq_len, hidden = output.shape
    output = output.view(bsz, seq_len, 32, (hidden//32)).transpose(1, 2).squeeze(0)
    collected_q_outputs.append(output.detach().cpu())

for target_length in seq_lengths:
    

    prompt = prompts[target_length]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len_1 = inputs["input_ids"].shape[1]
    print(seq_len_1)

    # cache = {}
    hooks_k = []
    hooks_q = []
    target_layer = 30
    layer = model.model.layers[target_layer].self_attn
    # print(f"  - K/V heads: {layer.config.num_key_value_heads}")
    
    
    # Register forward hooks
    hook_q = layer.q_proj.register_forward_hook(q_proj_hook)
    hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
    # hook_v = layer.q_proj.register_forward_hook(q_proj_hook)
    
    hooks_k.append(hook_k)
    hooks_q.append(hook_q)

    with torch.no_grad():
        outputs = model(**inputs)
    for hook in hooks_k:
        hook.remove()
    for hook in hooks_q:
        hook.remove()

    del inputs, outputs, prompt
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    # Q_list.append(Q)
    # K_list.append(K)

print(len(collected_q_outputs))

dot_means = []

for q, k in zip(collected_q_outputs, collected_k_outputs):
    num_q_heads, T, D = q.shape
    num_k_heads = k.shape[0]
    group_size = num_q_heads // num_k_heads
    print("group_size: ", group_size)

    assert num_q_heads % num_k_heads == 0, "Q/K head mismatch"

    # 每个 q head 匹配对应的 k head
    sim_all = []
    for qid in range(num_q_heads):
        kid = qid // group_size
        q_h = q[qid]          # [T, D]
        k_h = k[kid]          # [T, D]
        sim = torch.matmul(q_h, k_h.transpose(0, 1))  # [T, T]
        sim_all.append(sim.mean().item())             # mean over all entries

    dot_means.append(sum(sim_all) / len(sim_all))  # mean over all Q heads

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, dot_means, marker='o', label='Mean QK Dot-product (GQA)')
plt.xscale("log")
plt.xlabel("Sequence Length")
plt.ylabel("Mean Dot-product Similarity")
plt.title("QK Dot-product Similarity vs. Sequence Length (GQA/LLaMA3)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.title(f"K Projection (Head 0, Layer{30})")
plt.savefig(f"/home/azzhang/QK_analysis/qk_cone/layer{30}_K_Q_product.png")


