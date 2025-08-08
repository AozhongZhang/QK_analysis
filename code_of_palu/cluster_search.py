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
    with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompt_2k = prompts["2k"]
    inputs = tokenizer(prompt_2k, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    for hook in hooks_k:
        hook.remove()
    for hook in hooks_v:
        hook.remove()
    
    return collected_k_outputs, collected_v_outputs

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
    cka_matrix = torch.zeros(num_heads, num_heads)

    for i in range(num_heads):
        for j in range(num_heads):
            vi = matrix[i]
            vj = matrix[j]
            assert vi.shape == vj.shape
            vi = vi.cuda().float()
            vj = vj.cuda().float()
    
            cka_matrix[i, j] = linear_cka_centered_torch(vi, vj)
            
    return cka_matrix



def get_per_layer_group_to_rows(K_list_all_layers, compute_similarity, threshold=0.9):
    """
    Args:
        K_list_all_layers: list of [num_head, seq_len, head_dim] tensors, one per layer
        compute_similarity: function that returns CKA matrix of shape [num_head, num_head]
        threshold: float, similarity threshold for grouping

    Returns:
        group_to_rows_all_layers: dict of layer_idx -> group_to_rows dict
    """
    group_to_rows_all_layers = {}

    for layer_idx, K in enumerate(K_list_all_layers):
        cka_matrix = compute_similarity(K)  # shape [num_head, num_head]
        num_heads = cka_matrix.shape[0]
        
        group_to_rows = {}
        used = set()
        gid = 0
        for i in range(num_heads):
            if i in used:
                continue
            group_to_rows[gid] = [i]
            used.add(i)
            for j in range(i + 1, num_heads):
                if j not in used and cka_matrix[i, j] >= threshold:
                    group_to_rows[gid].append(j)
                    used.add(j)
            gid += 1
        
        group_to_rows_all_layers[layer_idx] = group_to_rows

    return group_to_rows_all_layers

# import numpy as np
# from sklearn.cluster import KMeans

# def get_per_layer_group_to_rows_by_cluster_center(K_list_all_layers, compute_similarity, group_size=4):
#     """
#     基于 head 的相似度向量进行 KMeans 聚类，再将 cluster 作为 group。
    
#     Args:
#         K_list_all_layers: list of [num_head, seq_len, head_dim] tensors, one per layer
#         compute_similarity: function that returns CKA matrix of shape [num_head, num_head]
#         group_size: int, 每个 group 的期望大小

#     Returns:
#         group_to_rows_all_layers: dict of layer_idx -> group_to_rows dict
#     """
#     group_to_rows_all_layers = {}

#     for layer_idx, K in enumerate(K_list_all_layers):
#         cka_matrix = compute_similarity(K)  # shape [num_heads, num_heads]
#         num_heads = cka_matrix.shape[0]

#         # 每个 head 的特征就是它与其它 head 的相似度向量
#         head_features = cka_matrix

#         # 目标分成的 group 数
#         n_groups = num_heads // group_size
#         print(n_groups)

#         # 用 KMeans 聚类
#         kmeans = KMeans(n_clusters=n_groups, random_state=0, n_init=10)
#         cluster_ids = kmeans.fit_predict(head_features)  # shape [num_heads]

#         # 构造 group_to_rows
#         group_to_rows = {}
#         for head_idx, cid in enumerate(cluster_ids):
#             group_to_rows.setdefault(cid, []).append(head_idx)

#         # 如果某些 group size 太大（> group_size），可以进行均匀再分组
#         final_group_to_rows = {}
#         gid = 0
#         for head_list in group_to_rows.values():
#             for i in range(0, len(head_list), group_size):
#                 chunk = head_list[i:i+group_size]
#                 if len(chunk) == group_size:  # 保证 group size 一致（可选）
#                     final_group_to_rows[gid] = chunk
#                     gid += 1

#         group_to_rows_all_layers[layer_idx] = final_group_to_rows

#     return group_to_rows_all_layers


# # import torch
# from collections import defaultdict

# def group_heads_by_cka_matrix(K, cka_matrix, threshold=0.7):
#     """
#     K_list: list of tensors of shape [num_head, seq_len, head_dim]
#     compute_similarity: function that returns [num_head, num_head] CKA matrix
#     threshold: similarity threshold for grouping

#     Returns:
#         group_to_rows: dict like {0: [0, 3], 1: [1, 2, 5], ...}
#     """
#     # Step 1: Concatenate K/V across batch dim
#     # K_all = torch.cat(K_list, dim=1)  # [num_head, total_seq_len, head_dim]

#     # Step 2: Compute CKA similarity matrix
#     # cka_matrix = compute_similarity(K)  # [num_head, num_head]
#     num_heads = cka_matrix.shape[0]

#     # Step 3: Greedy grouping
#     group_to_rows = defaultdict(list)
#     used = set()
#     gid = 0

#     for i in range(num_heads):
#         if i in used:
#             continue
#         group_to_rows[gid].append(i)
#         used.add(i)
#         for j in range(i + 1, num_heads):
#             if j not in used and cka_matrix[i, j] >= threshold:
#                 group_to_rows[gid].append(j)
#                 used.add(j)
#         gid += 1

#     return dict(group_to_rows)

# test
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# K, V = collect_headwise_KV(model, tokenizer, model.device)

# # result = get_per_layer_group_to_rows(K, compute_similiarity, 0.7)
# result = get_per_layer_group_to_rows_by_cluster_center(K, compute_similiarity)
# # cka_sim = compute_similiarity(K[0])
# # result = group_heads_by_cka_matrix(K[0], cka_sim)
# print(result)