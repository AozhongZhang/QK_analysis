import os, click
import torch
import torch.nn as nn
from loguru import logger
# from .model import AVAILABLE_MODELS
# from .data_utils import get_calib_data
import math
from tqdm import tqdm
import json


def collect_headwise_KV(model, tokenizer, device):
    
    # calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)
    
    # We'll collect outputs for all layers in these lists
    collected_k_outputs = []
    collected_v_outputs = []
    
    def k_proj_hook(module, input, output):
        """
        module: The layer that produced this output (k_proj).
        input:  The input to k_proj.
        output: The output from k_proj (shape [batch_size, seq_len, hidden_dim]).
        """
        B, S, D = output.shape
        H = 32
        head_dim = D // H
        output = output.view(B, S, H, head_dim).transpose(1, 2).squeeze(0) # [num_head, seq_len, head_dim]
        # Detach to avoid growing the autograd graph
        collected_k_outputs.append(output.detach().cpu())

    def v_proj_hook(module, input, output):
        """
        module: The layer that produced this output (v_proj).
        input:  The input to v_proj.
        output: The output from v_proj (shape [batch_size, seq_len, hidden_dim]).
        """
        B, S, D = output.shape
        H = 32
        head_dim = D // H
        output = output.view(B, S, H, head_dim).transpose(1, 2).squeeze(0) # [num_head, seq_len, head_dim]
        # Detach to avoid growing the autograd graph
        collected_v_outputs.append(output.detach().cpu())

    num_layers = len(model.model.layers)
    hooks_k = []
    hooks_v = []
    for layer_idx in range(num_layers):
        # Access the i-th layer
        layer = model.model.layers[layer_idx].self_attn
        # print(f"  - K/V heads: {layer.config.num_key_value_heads}")
        
        
        # Register forward hooks
        hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
        hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
        
        hooks_k.append(hook_k)
        hooks_v.append(hook_v)


    model.eval()
    model.to(device)
    with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama2.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompt_4k = prompts["4k"]
    inputs = tokenizer(prompt_4k, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    for hook in hooks_k:
        hook.remove()
    for hook in hooks_v:
        hook.remove()
    
    return collected_k_outputs, collected_v_outputs

def collect_W_kv(model):
    num_layers = len(model.model.layers)
    W_k = []
    W_v = []
    num_heads = 32
    for layer_idx in range(num_layers):
        # Access the i-th layer
        layer = model.model.layers[layer_idx].self_attn
        # print(f"  - K/V heads: {layer.config.num_key_value_heads}")
        w_k_weight = layer.k_proj.weight.data.clone()
        w_v_weight = layer.v_proj.weight.data.clone()
        # print(w_k_weight.shape)
        
        w_k = w_k_weight.view(num_heads, -1, w_k_weight.shape[1])
        w_v = w_v_weight.view(num_heads, -1, w_v_weight.shape[1])
        # print(w_k.shape)
        W_k.append(w_k)
        W_v.append(w_v)

    return W_k, W_v

def linear_cka_centered_torch(kv1: torch.Tensor, kv2: torch.Tensor) -> torch.Tensor:
    """
    A *centered* linear CKA, as in Kornblith et al. (2019), for (L, D) Tensors.
    This subtracts each row's mean from kv1, kv2 before computing the norm-based formula.
    
    Steps:
      1. Row-center each representation (i.e., subtract column means).
      2. Compute Frobenius norms of X^T X, Y^T Y, X^T Y on the centered data.
      3. Return (||X^T Y||_F^2) / (||X^T X||_F * ||Y^T Y||_F).

    Note:
      - 'Row-center' means we subtract the *column* mean for each dimension (the usual approach 
        in CKA references). This ensures the average vector over all tokens is zero.

    Args:
      kv1: shape (L, D)
      kv2: shape (L, D)

    Returns:
      cka_value: a scalar torch.Tensor
    """
    assert kv1.shape[1] == kv2.shape[1], "kv1, kv2 must have same embedding dimension."

    # Move to GPU if desired
    device = kv1.device
    kv1 = kv1.to(device)
    kv2 = kv2.to(device)
    
    # 1. Row-center each representation. 
    #    (Compute column means & subtract => each dimension has mean 0 across L)
    kv1_centered = kv1 - kv1.mean(dim=0, keepdim=True)
    kv2_centered = kv2 - kv2.mean(dim=0, keepdim=True)
    
    # 2. Norm computations
    xtx = (kv1_centered.T @ kv1_centered).norm(p='fro')
    yty = (kv2_centered.T @ kv2_centered).norm(p='fro')
    xty = (kv1_centered.T @ kv2_centered).norm(p='fro')

    # Handle degenerate case
    if xtx == 0 or yty == 0:
        return torch.tensor(0.0, device=device, dtype=kv1.dtype)

    # 3. Linear CKA formula
    cka_value = (xty ** 2) / (xtx * yty)

    return cka_value

def compute_similiarity(matrix):
    
    num_heads, seq_len, head_dim = matrix.shape
    cka_matrix = torch.zeros(num_heads, num_heads)
    # cka_matrix = torch.zeros(num_heads, num_heads)

    for i in range(num_heads):
        for j in range(num_heads):
            vi = matrix[i]
            vj = matrix[j]
            assert vi.shape == vj.shape
            vi = vi.cuda().float()
            vj = vj.cuda().float()
    
            cka_matrix[i, j] = linear_cka_centered_torch(vi, vj)
            
    return cka_matrix


import torch
from typing import Dict, List, Tuple

def cka_grouping_by_max_similarity(cka_matrix: torch.Tensor, threshold: float = 0.7) -> Dict[int, List[int]]:
    num_heads = cka_matrix.size(0)
    grouped = set()
    group_to_rows = {}
    head_to_group = {}
    gid = 0

    # Step 1: 将所有最大相似度小于 threshold 的 head 单独成组
    for i in range(num_heads):
        max_sim = max([cka_matrix[i][j].item() for j in range(num_heads) if j != i])
        if max_sim < threshold:
            group_to_rows[gid] = [i]
            head_to_group[i] = gid
            grouped.add(i)
            gid += 1

    # Step 2: 计算所有 head pair 相似度，按从高到低排序
    pair_sims = []
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            if i in grouped or j in grouped:
                continue
            sim = cka_matrix[i][j].item()
            pair_sims.append(((i, j), sim))
    pair_sims.sort(key=lambda x: -x[1])  # 降序排列

    for (i, j), sim in pair_sims:
        if i in grouped and j in grouped:
            continue
        elif i in grouped:
            group_id = head_to_group[i]
            if j not in grouped:
                group_to_rows[group_id].append(j)
                head_to_group[j] = group_id
                grouped.add(j)
        elif j in grouped:
            group_id = head_to_group[j]
            if i not in grouped:
                group_to_rows[group_id].append(i)
                head_to_group[i] = group_id
                grouped.add(i)
        elif i not in grouped and j not in grouped:
            group_to_rows[gid] = [i, j]
            head_to_group[i] = gid
            head_to_group[j] = gid
            grouped.add(i)
            grouped.add(j)
            gid += 1

    return group_to_rows


from typing import Dict, List
import torch
import numpy as np

def cka_grouping_fixed_num(
    cka_matrix: torch.Tensor,
    num_groups: int,
    threshold: float = 0.7
) -> Dict[int, List[int]]:
    num_heads = cka_matrix.size(0)
    grouped = set()
    group_to_rows = {}
    head_to_group = {}
    gid = 0

    # Step 1: 尝试独立分组低相似度 head
    singleton_heads = []
    for i in range(num_heads):
        max_sim = max([cka_matrix[i][j].item() for j in range(num_heads) if j != i])
        if max_sim < threshold:
            singleton_heads.append(i)

    if len(singleton_heads) >= num_groups:
        # 如果低相似度 head 太多，优先保留前 num_groups 个为单独组
        for i in singleton_heads[:num_groups]:
            group_to_rows[gid] = [i]
            gid += 1
        return group_to_rows

    # Step 2: 分配这些低相似度 head 为单独组
    for i in singleton_heads:
        group_to_rows[gid] = [i]
        head_to_group[i] = gid
        grouped.add(i)
        gid += 1

    # Step 3: 对剩下的 head 进行聚类，分配剩余的组数
    remaining_heads = [i for i in range(num_heads) if i not in grouped]
    remaining_group_budget = num_groups - gid

    if remaining_group_budget <= 0:
        return group_to_rows

    # Step 4: 聚类剩余 head（简单 KMeans 风格，基于相似度）
    from sklearn.cluster import KMeans

    cka_numpy = cka_matrix.detach().cpu().numpy()
    # 用每个 head 的 CKA 向量作为特征（也可做 PCA 降维）
    head_features = cka_numpy[remaining_heads][:, remaining_heads]
    
    # k-means 聚类
    km = KMeans(n_clusters=remaining_group_budget, random_state=42).fit(head_features)
    labels = km.labels_
    
    for i, label in enumerate(labels):
        h = remaining_heads[i]
        real_gid = gid + label
        if real_gid not in group_to_rows:
            group_to_rows[real_gid] = []
        group_to_rows[real_gid].append(h)

    return group_to_rows



import torch
import numpy as np
from typing import Dict, List, Tuple

def greedy_group_by_cka(
    cka_matrix: torch.Tensor,
    group_size: int = 4,
    seed: int = 42
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    使用贪心算法基于 CKA 相似度将 head 重新排序并划分 group。
    """
    rng = np.random.default_rng(seed)
    cka = cka_matrix.cpu().numpy()
    num_heads = cka.shape[0]

    remaining = set(range(num_heads))
    group_to_rows: Dict[int, List[int]] = {}
    order: List[int] = []

    gid = 0
    while remaining:
        # 1. 选一个随机种子 head
        seed_head = rng.choice(list(remaining))
        remaining.remove(seed_head)

        # 2. 找出与 seed 最相似的 (group_size - 1) 个 head
        sim = cka[seed_head].copy()
        sim[list(set(range(num_heads)) - remaining - {seed_head})] = -np.inf
        top_indices = np.argsort(sim)[- (group_size - 1):][::-1]
        top_heads = []

        for idx in top_indices:
            if idx in remaining:
                top_heads.append(idx)
                remaining.remove(idx)
            if len(top_heads) == group_size - 1:
                break

        # 3. 如果还不够，继续从剩下里随机补
        while len(top_heads) < group_size - 1 and remaining:
            extra = rng.choice(list(remaining))
            top_heads.append(extra)
            remaining.remove(extra)

        # 4. 组成组并记录
        group = [seed_head] + top_heads
        group_to_rows[gid] = group
        order.extend(group)
        gid += 1

    return group_to_rows

import itertools

def greedy_group_by_cka_strict(
    cka_matrix: torch.Tensor,
    group_size: int = 4,
    seed: int = 42
) -> Tuple[Dict[int, List[int]], List[int]]:
    rng = np.random.default_rng(seed)
    cka = cka_matrix.cpu().numpy()
    num_heads = cka.shape[0]

    remaining = set(range(num_heads))
    group_to_rows: Dict[int, List[int]] = {}
    order: List[int] = []
    gid = 0

    while len(remaining) >= group_size:
        best_group = None
        best_score = -np.inf

        # 遍历所有可能的组合
        for combo in itertools.combinations(remaining, group_size):
            score = 0.0
            count = 0
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    score += cka[combo[i], combo[j]]
                    count += 1
            avg_score = score / count

            if avg_score > best_score:
                best_score = avg_score
                best_group = combo

        if best_group is not None:
            group_to_rows[gid] = list(best_group)
            order.extend(best_group)
            remaining.difference_update(best_group)
            gid += 1
        else:
            break

    # 将剩余未分组的 head 单独成组或加入已有组
    if remaining:
        leftovers = list(remaining)
        for head in leftovers:
            group_to_rows[gid] = [head]
            order.append(head)
            gid += 1

    return group_to_rows

import torch
import numpy as np
from typing import Dict, List, Tuple

def fast_pairwise_group_by_cka(
    cka_matrix: torch.Tensor,
    group_size: int = 4,
    sim_threshold: float = 0.5,
    seed: int = 42
) -> Tuple[Dict[int, List[int]], List[int]]:
    rng = np.random.default_rng(seed)
    cka = cka_matrix.cpu().numpy()
    num_heads = cka.shape[0]
    remaining = set(range(num_heads))
    group_to_rows: Dict[int, List[int]] = {}
    order: List[int] = []
    gid = 0

    # 计算每个 head 的平均相似度（可用于优先排序）
    avg_sim = np.sum(cka, axis=1)
    sorted_heads = np.argsort(-avg_sim)  # 从相似度大的先开始

    for head in sorted_heads:
        if head not in remaining:
            continue

        # 从剩下中选出最相似的 group_size - 1 个
        sim = cka[head]
        candidates = [(i, sim[i]) for i in remaining if i != head]
        candidates.sort(key=lambda x: -x[1])

        group = [head]
        for i, _ in candidates:
            # 检查是否与组内已有的所有 head 相似
            if all(cka[i][j] >= sim_threshold for j in group):
                group.append(i)
            if len(group) == group_size:
                break

        if len(group) == group_size:
            for h in group:
                remaining.remove(h)
            group_to_rows[gid] = group
            order.extend(group)
            gid += 1

    # 将剩下的随便分组填满（不保证组内相似）
    remaining = list(remaining)
    for i in range(0, len(remaining), group_size):
        group = remaining[i:i+group_size]
        group_to_rows[gid] = group
        order.extend(group)
        gid += 1

    return group_to_rows



def reindex_group_ids(group_to_rows: Dict[int, List[int]]) -> Dict[int, List[int]]:
    new_group_to_rows = {}
    for new_gid, old_gid in enumerate(sorted(group_to_rows.keys())):
        new_group_to_rows[new_gid] = group_to_rows[old_gid]
    return new_group_to_rows

def get_group_to_rows_all_layers_by_cka(K_list_all_layers, compute_cka_fn):
    all_group_to_rows = {}
    for layer_idx, K in enumerate(K_list_all_layers):
        cka = compute_cka_fn(K)
        group_to_rows = greedy_group_by_cka(cka, group_size=8)
        # group_to_rows = fast_pairwise_group_by_cka(cka, group_size=4)
        # group_to_rows = cka_grouping_by_max_similarity(cka, 0.5)
        # group_to_rows = cka_grouping_fixed_num(cka, 4, 0.5)
        # group_to_rows = reindex_group_ids(group_to_rows)
        all_group_to_rows[layer_idx] = group_to_rows
    return all_group_to_rows


def build_ordered_group_to_rows(K_list_all_layers, group_size=32):
    """
    Args:
        K_list_all_layers: List of tensors, each with shape [num_head, seq_len, head_dim]
        group_size: fixed size for each group

    Returns:
        group_to_rows_all_layers: Dict[int, Dict[int, List[int]]], layer_idx -> group_id -> list of head idx
    """
    group_to_rows_all_layers = {}

    for layer_idx, K in enumerate(K_list_all_layers):
        num_heads = K.shape[0]
        group_to_rows = {}
        gid = 0
        for i in range(0, num_heads, group_size):
            group_to_rows[gid] = list(range(i, min(i + group_size, num_heads)))
            gid += 1
        group_to_rows_all_layers[layer_idx] = group_to_rows

    return group_to_rows_all_layers

import torch
from typing import Dict, List





# # test
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# K, V = collect_headwise_KV(model, tokenizer, model.device)
# result = build_ordered_group_to_rows(K)
# print(result)
# print(K[0].shape)

# # result = get_per_layer_group_to_rows(K, compute_similiarity, 0.7)
# result = get_per_layer_group_to_rows_by_cluster_center(K, compute_similiarity)
# cka_sim = compute_similiarity(K[0])
# print(cka_sim)
# result = cka_grouping_by_max_similarity(cka_sim)
# result, order = cka_grouping_auto_group_count(cka_sim)
# print(result)
# # result = group_heads_by_cka_matrix(K[0], cka_sim)
# cka_matrix_K0 = compute_similiarity(K[0])
# print(cka_matrix_K0)
# result, order = greedy_group_by_cka(cka_matrix_K0)
# result = get_group_to_rows_all_layers_by_cka(K, compute_similiarity)
# result, order = reorder_heads_by_cka(cka_matrix_K0, 4)
# print(cka_matrix_K0)
# result = auto_cluster_heads(K[0])
# result = auto_cluster_heads_from_cka(cka_matrix_K0)
# result = get_group_to_rows_all_layers_by_cka(K, compute_similiarity)

# print(result)
# print(order)

# import matplotlib.pyplot as plt
# import seaborn as sns

# reordered = cka_matrix_K0[order][:, order]
# plt.figure(figsize=(6, 5))
# sns.heatmap(reordered.cpu().numpy(), cmap="coolwarm", vmin=0, vmax=1)
# plt.title("CKA Similarity after Reordering")
# plt.xlabel("Head Index")
# plt.ylabel("Head Index")
# plt.grid(True)
# plt.tight_layout()

    
# plt.savefig(f"/home/azzhang/Palu/palu/raw_rope.png")
# plt.show()

# def plot_reordered_cka(cka_matrix, order):
#     reordered = cka_matrix[order][:, order]
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(reordered.cpu().numpy(), cmap="coolwarm", vmin=0, vmax=1)
#     plt.title("CKA Similarity after Reordering")
#     plt.xlabel("Head Index")
#     plt.ylabel("Head Index")
#     plt.show()